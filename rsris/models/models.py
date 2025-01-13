import math
from typing import Optional, List, Tuple
import einops
from peft import LoraConfig, get_peft_model
from torchvision.transforms.functional import normalize
import torch
from mmcv.cnn import ConvModule, build_norm_layer
from modelscope import snapshot_download
from torch import Tensor, nn
from transformers.models.sam.modeling_sam import SamVisionEncoder
from mmengine.model import BaseModule
from transformers import AutoConfig, AutoModel, SamConfig, SamModel, SiglipModel, SamProcessor, SiglipProcessor, \
    AutoTokenizer
from mmseg.models import EncoderDecoder, accuracy
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, OptConfigType, OptMultiConfig, SampleList, MultiConfig
import torch.nn.functional as F


@MODELS.register_module()
class RefSegEncoderDecoder(EncoderDecoder):
    def __init__(
            self,
            lora_cfg: dict,
            clip_vision_encoder: str,
            clip_text_encoder: str,
            sam_prompt_encoder: str,
            sam_mask_decoder: str,
            down_spatial_times: int = 2,
            with_dense_feat: bool = True,
            *args,
            norm_cfg: ConfigType = dict(type='SyncBN', requires_grad=True),
            act_cfg: ConfigType = dict(type='GELU'),
            **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_cfg = lora_cfg
        self.clip_vision_encoder = MODELS.build(clip_vision_encoder)
        self.clip_text_encoder = MODELS.build(clip_text_encoder)
        self.sam_prompt_encoder = MODELS.build(sam_prompt_encoder)
        self.sam_mask_decoder = MODELS.build(sam_mask_decoder)
        self.down_spatial_times = down_spatial_times
        self.with_dense_feat = with_dense_feat

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg


        self.prompter_down_channel = ConvModule(
                in_channels=self.clip_text_encoder.config.hidden_size*3,
                out_channels=self.sam_prompt_encoder.hidden_size,
                kernel_size=1,
                stride=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.prompter_down_spatial = nn.Sequential(*[
            ConvModule(
                in_channels=self.sam_prompt_encoder.hidden_size,
                out_channels=self.sam_prompt_encoder.hidden_size,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg) for _ in range(down_spatial_times)],
            ConvModule(
                in_channels=self.sam_prompt_encoder.hidden_size,
                out_channels=self.sam_prompt_encoder.hidden_size,
                kernel_size=1,
                stride=1,
                norm_cfg=None,
                act_cfg=None)
        )

        # set efficient finetuning parameters
        self.set_finetune_parameters()
        self.print_trainable_parameters()


    def set_finetune_parameters(self):
        # SAM vision encoder: backbone
        # CLIP vision encoder: clip_vision_encoder
        # CLIP text encoder: clip_text_encoder
        # SAM prompt encoder: sam_prompt_encoder
        # SAM mask decoder: sam_mask_decoder
        # prompter_down_channel: prompter_down_channel
        # prompter_down_spatial: prompter_down_spatial
        peft_keys = ['backbone', 'clip_vision_encoder', 'clip_text_encoder']
        for k in peft_keys:
            if k in self.lora_cfg:
                v_ = self.lora_cfg[k].copy()
                lora_config = LoraConfig(**v_)
                setattr(self, k, get_peft_model(getattr(self, k), lora_config))
            else:
                print(f"Warning: {k} not in lora_cfg")
                getattr(self, k).requires_grad_(False)
                print(f"Set {k} to requires_grad=False")

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param


    def print_trainable_parameters(self) -> None:
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )


    def get_image_positional_embeddings(self, size):
        target_device = self.backbone.shared_image_embedding.positional_embedding.device
        target_dtype = self.backbone.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.backbone.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    def extract_feat(self, inputs, text_list) -> List[Tensor]:

        # for SAM vision encoder
        x_sam = normalize(inputs, mean=self.backbone.processor.image_mean, std=self.backbone.processor.image_std)
        sam_visual_feat = self.backbone(x_sam)
        sam_visual_feat = sam_visual_feat['last_hidden_state']  # BX256X64X64

        # for CLIP vision encoder
        x_clip = normalize(inputs, mean=self.clip_vision_encoder.processor.image_mean, std=self.clip_vision_encoder.processor.image_std)
        x_clip = F.interpolate(x_clip, size=list(self.clip_vision_encoder.processor.size.values()), mode='bilinear', align_corners=False)
        clip_visual_feat = self.clip_vision_encoder(x_clip)
        clip_visual_feat_pooler = clip_visual_feat['pooler_output']  # BX1152
        clip_visual_feat = clip_visual_feat['last_hidden_state'] # BX729X1152
        clip_visual_feat = einops.rearrange(clip_visual_feat, 'b (h w) c -> b c h w', h=int(math.sqrt(clip_visual_feat.shape[1])))  # BX1152X27X27

        text_dict = self.clip_text_encoder.processor(text_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
        text_dict = {k: v.to(inputs.device) for k, v in text_dict.items()}
        input_ids = text_dict['input_ids']
        clip_text_feat = self.clip_text_encoder(**text_dict)
        clip_text_feat_pooler = clip_text_feat['pooler_output']  # BX1152
        clip_text_feat = clip_text_feat['last_hidden_state']  # BX7X1152

        normalized_clip_visual_feat = clip_visual_feat / clip_visual_feat.norm(p=2, dim=1, keepdim=True)  # BX1152X27X27
        normalized_clip_text_feat = clip_text_feat / clip_text_feat.norm(p=2, dim=2, keepdim=True) # BX7X1152
        normalized_clip_text_feat_pooler = clip_text_feat_pooler / clip_text_feat_pooler.norm(p=2, dim=1, keepdim=True) # BX1152

        local_activate = einops.einsum(normalized_clip_visual_feat, normalized_clip_text_feat, 'b c h w, b d c -> b d h w')
        local_activate = local_activate * self.clip_text_encoder.logit_scale.exp() + self.clip_text_encoder.logit_bias
        local_activate = F.sigmoid(local_activate) # BX7X27X27
        local_activated_feat = einops.einsum(local_activate, clip_visual_feat, 'b d h w, b c h w -> b c h w') / local_activate.size(1) # BX1152X27X27
        local_clip_visual_feat = (clip_visual_feat + local_activated_feat) / 2

        global_activate = einops.einsum(normalized_clip_visual_feat, normalized_clip_text_feat_pooler, 'b c h w, b c -> b h w')
        global_activate_logit = global_activate * self.clip_text_encoder.logit_scale.exp() + self.clip_text_encoder.logit_bias
        global_activate = F.sigmoid(global_activate_logit)  # BX27X27
        global_activated_feat = einops.einsum(global_activate, clip_visual_feat, 'b h w, b c h w -> b c h w') # BX1152X27X27


        clip_activated_feat = torch.cat([local_clip_visual_feat, global_activated_feat, clip_visual_feat], dim=1)
        clip_activated_feat = self.prompter_down_channel(clip_activated_feat)
        clip_activated_feat = clip_activated_feat + self.get_image_positional_embeddings(clip_activated_feat.size(2))
        clip_activated_feat = self.prompter_down_spatial(clip_activated_feat) # BX768X7X7

        # repeat with batch size
        batch_size = inputs.size(0)
        image_positional_embeddings = self.get_image_positional_embeddings(sam_visual_feat.size(2))
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        if self.with_dense_feat:
            global_activate_logit = F.interpolate(global_activate_logit.unsqueeze(1), size=(image_positional_embeddings.size(2)*4, image_positional_embeddings.size(3)*4), mode='bilinear', align_corners=False)
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=global_activate_logit,
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(
                input_points=None,
                input_labels=None,
                input_boxes=None,
                input_masks=None,
            )

        sparse_embeddings = einops.rearrange(clip_activated_feat, 'b c h w -> b 1 (h w) c')

        low_res_masks, iou_predictions, mask_decoder_attentions = self.sam_mask_decoder(
            image_embeddings=sam_visual_feat,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            attention_similarity=None,
            target_embedding=None,
            output_attentions=None,
        )
        seg_mask = low_res_masks.squeeze(1)
        return seg_mask

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        text_list = [data_sample.get('text', '') for data_sample in data_samples]
        x = self.extract_feat(inputs, text_list)

        losses = dict()
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self, inputs: Tensor, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        batch_img_metas_with_text = []
        for img_meta, data_sample in zip(batch_img_metas, data_samples):
            img_meta['text'] = data_sample.get('text', '')
            batch_img_metas_with_text.append(img_meta)
        seg_logits = self.inference(inputs, batch_img_metas_with_text)

        return self.postprocess_result(seg_logits, data_samples)

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        text_list = [img_meta['text'] for img_meta in batch_img_metas]
        x = self.extract_feat(inputs, text_list)
        seg_logits = self.decode_head.predict(x, batch_img_metas, self.test_cfg)
        return seg_logits



@MODELS.register_module()
class RefSegSiglipTextModel(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='thomas/siglip-so400m-patch14-384',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        self.processor = SiglipProcessor.from_pretrained(model_name_or_path).tokenizer
        model = SiglipModel.from_pretrained(model_name_or_path)
        self.model = model.text_model
        self.config = self.model.config
        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias
        self.model.is_init = True
        self.logit_scale.is_init = True
        self.logit_bias.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        results = self.model(*args, **kwargs)
        return results


@MODELS.register_module()
class RefSegSamVisionEncoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        self.processor = SamProcessor.from_pretrained(model_name_or_path).image_processor
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.vision_encoder
        self.config = self.model.config
        self.shared_image_embedding = model.shared_image_embedding
        self.model.is_init = True
        self.shared_image_embedding.is_init = True


    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@MODELS.register_module()
class RefSegSamPromptEncoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str = 'KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.prompt_encoder
        self.hidden_size = self.model.hidden_size
        self.model.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@MODELS.register_module()
class RefSegSamMaskDecoder(BaseModule):
    def __init__(
            self,
            model_name_or_path: str = 'KyanChen/sam-vit-base',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        model = SamModel.from_pretrained(model_name_or_path)
        self.model = model.mask_decoder
        self.hidden_size = self.model.hidden_size
        self.model.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@MODELS.register_module()
class RefSegSiglipVisionModel(BaseModule):
    def __init__(
            self,
            model_name_or_path: str='thomas/siglip-so400m-patch14-384',
            init_cfg=None
    ):
        super().__init__(init_cfg=init_cfg)
        self.model_name_or_path = model_name_or_path
        model_name_or_path = snapshot_download(model_name_or_path)
        self.processor = SiglipProcessor.from_pretrained(model_name_or_path).image_processor
        self.model = SiglipModel.from_pretrained(model_name_or_path).vision_model
        self.config = self.model.config
        self.model.is_init = True

    def init_weights(self):
        pass

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)



@MODELS.register_module()
class PseudoSegHead(BaseModule):
    def __init__(
            self,
            num_classes: int=2,
            out_channels: int=1,
            threshold: float=0.5,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            ignore_index=255,
            sampler=None,
            align_corners=False,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.out_channels = out_channels
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.align_corners = align_corners

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(f'loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}')

        if sampler is not None:
            self.sampler = build_pixel_sampler(sampler, context=self)
        else:
            self.sampler = None

    def forward(self, inputs):
        return inputs

    def loss(self, inputs, batch_data_samples: SampleList, train_cfg: ConfigType) -> dict:
        seg_logits = self.forward(inputs)
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict], test_cfg: ConfigType) -> Tensor:
        seg_logits = self.forward(inputs)
        return self.predict_by_feat(seg_logits, batch_img_metas)

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:
        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits




if __name__ == '__main__':
    sam_vision_encoder = RefSegSamVisionEncoder(model_name_or_path='KyanChen/sam-vit-base')
    print(sam_vision_encoder.model.is_init)