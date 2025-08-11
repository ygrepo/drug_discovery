import torch
import torch.nn as nn

from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM, EsmTokenizer
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
from src.modeling_esm import EsmForMaskedLM, EsmForMutationDesign


class MutaPLM(nn.Module):
    def __init__(
        self,
        protein_model=None,
        llama_ckpt=None,
        llama_pretrained_ckpt=None,
        num_query_tokens_protein1=64,
        num_query_tokens_protein2=64,
        ca_num_head=8,
        protein_maxlen=1024,
        text_maxlen=256,
        func_maxlen=512,
        test_mode=False,
        resume=False,
        device=None,
        m2t=True,
        t2m=True,
        pretrain=False,
    ):
        super(MutaPLM, self).__init__()
        self.device = device
        self.num_query_tokens_protein1 = num_query_tokens_protein1
        self.num_query_tokens_protein2 = num_query_tokens_protein2
        self.ca_num_head = ca_num_head
        self.protein_maxlen = protein_maxlen
        self.text_maxlen = text_maxlen
        self.func_maxlen = func_maxlen
        self.m2t = m2t
        self.t2m = t2m
        self.pretrain = pretrain

        # load esm
        print("*** loading protein model...")
        if pretrain:
            self.protein_model = EsmForMaskedLM.from_pretrained(
                protein_model, torch_dtype=torch.bfloat16
            )
            self.forward_fn = self.forward_pt
            self.loss_names = []
            if self.m2t:
                self.loss_names.append("loss_p2t")
            if self.t2m:
                self.loss_names.append("loss_t2p")
        else:
            self.protein_model = EsmForMutationDesign.from_pretrained(
                protein_model, torch_dtype=torch.bfloat16
            )  # delta decoder is here
            self.forward_fn = self.forward_ft
            self.loss_names = []
            if self.m2t:
                self.loss_names.append("loss_m2t")
            if self.t2m:
                self.loss_names += ["loss_pos", "loss_aa"]
        self.protein_tokenizer = EsmTokenizer.from_pretrained(protein_model)
        print("*** freezing protein model...")
        for name, param in self.protein_model.named_parameters():
            if (
                not "_adapter" in name
                and not "mutation_classifier" in name
                and not "lm_head" in name
            ):
                param.requires_grad = False

        # load llm
        print("*** loading llm tokenizer...")
        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            llama_ckpt, truncation_side="left"
        )
        self.llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.llm_tokenizer.add_special_tokens({"bos_token": "<s>"})
        self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})
        self.llm_tokenizer.add_special_tokens({"unk_token": "<unk>"})
        print(f"*** loading llm from {llama_ckpt}...")
        if pretrain:
            self.llm = LlamaForCausalLM.from_pretrained(
                llama_ckpt, torch_dtype=torch.bfloat16
            )
        else:
            cfg = LlamaConfig.from_pretrained(llama_ckpt)
            self.llm = LlamaForCausalLM(cfg)
        self.llm.resize_token_embeddings(len(self.llm_tokenizer))

        # add lora
        print("*** adding LoRA...")
        lora_config = LoraConfig(
            peft_type=TaskType.CAUSAL_LM,
            inference_mode=test_mode,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ],
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        # delta encoder with cross attention
        print("*** building delta encoder...")
        self.query_protein1 = nn.Parameter(
            torch.zeros(
                1, num_query_tokens_protein1, self.protein_model.config.hidden_size
            )
        )
        nn.init.normal_(self.query_protein1, 0, 0.02)
        self.query_protein2 = nn.Parameter(
            torch.zeros(
                1, num_query_tokens_protein2, self.protein_model.config.hidden_size
            )
        )
        nn.init.normal_(self.query_protein2, 0, 0.02)
        self.pooler_protein1 = nn.MultiheadAttention(
            embed_dim=self.protein_model.config.hidden_size,
            num_heads=self.ca_num_head,
            batch_first=True,
        )
        self.pooler_protein2 = nn.MultiheadAttention(
            embed_dim=self.protein_model.config.hidden_size,
            num_heads=self.ca_num_head,
            batch_first=True,
        )

        self.bop_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.eop_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.bom_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.eom_embeds = nn.Parameter(torch.zeros(1, 1, self.llm.config.hidden_size))
        self.soft_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens_protein2, self.llm.config.hidden_size)
        )
        nn.init.normal_(self.bop_embeds, 0, 0.02)
        nn.init.normal_(self.eop_embeds, 0, 0.02)
        nn.init.normal_(self.bom_embeds, 0, 0.02)
        nn.init.normal_(self.eom_embeds, 0, 0.02)
        nn.init.normal_(self.soft_tokens, 0, 0.02)

        # build proj
        self.proj_protein1 = nn.Linear(
            self.protein_model.config.hidden_size, self.llm.config.hidden_size
        )
        self.proj_protein2 = nn.Linear(
            self.protein_model.config.hidden_size, self.llm.config.hidden_size
        )
        self.proj_text = nn.Linear(
            self.llm.config.hidden_size, self.protein_model.config.hidden_size
        )

        if not pretrain and llama_pretrained_ckpt is not None:
            print(f"*** loading pretrained llm from {llama_pretrained_ckpt}...")
            ckpt = torch.load(llama_pretrained_ckpt, map_location=torch.device("cpu"))[
                "model"
            ]
            print(self.load_state_dict(self.convert_params(ckpt), strict=False))
            del ckpt

        if not m2t:
            print("*** freeze m2t parameters")
            self.freeze_m2t_params()
        print("*** model built successfully.")

    def freeze_m2t_params(self):
        for param in self.pooler_protein1.parameters():
            param.requires_grad = False
        for param in self.pooler_protein2.parameters():
            param.requires_grad = False
        for param in self.proj_protein1.parameters():
            param.requires_grad = False
        for param in self.proj_protein2.parameters():
            param.requires_grad = False
        self.query_protein1.requires_grad = False
        self.query_protein2.requires_grad = False
        self.bop_embeds.requires_grad = False
        self.eop_embeds.requires_grad = False
        self.bom_embeds.requires_grad = False
        self.eom_embeds.requires_grad = False

    def convert_params(self, ckpt):
        # Initialize parameters for fine-tuning
        # pooler_protein -> pooler_protein 1&2
        # query_protein -> query_protein 1&2
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            if "pooler_protein" in k:
                new_ckpt[k.replace("pooler_protein", "pooler_protein1")] = v
                new_ckpt[k.replace("pooler_protein", "pooler_protein2")] = v
            elif k.startswith("proj"):
                new_ckpt[k.replace("proj", "proj_protein1")] = v
                new_ckpt[k.replace("proj", "proj_protein2")] = v
            elif "query_protein" in k:
                new_ckpt[k.replace("query_protein", "query_protein1")] = v
                new_ckpt[k.replace("query_protein", "query_protein2")] = v
            elif "bop_embeds" in k:
                new_ckpt[k] = v
                new_ckpt[k.replace("bop_embeds", "bom_embeds")] = v
            elif "eop_embeds" in k:
                new_ckpt[k] = v
                new_ckpt[k.replace("eop_embeds", "eom_embeds")] = v
            else:
                new_ckpt[k] = v

        return new_ckpt

    def maybe_autocast(self, dtype=torch.bfloat16):
        if self.device.type in "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=dtype)
        elif self.device.type == "cpu":
            return torch.amp.autocast(device_type="cpu", dtype=dtype)
        else:
            return contextlib.nullcontext()

    def _encode_protein(self, protein1, protein2):
        batch_size = len(protein1)
        protein1 = self.protein_tokenizer(
            protein1,
            max_length=self.protein_maxlen,
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(
            self.device
        )  # input_ids: [bs, prot_len]
        if protein2 is not None:
            protein2 = self.protein_tokenizer(
                protein2,
                max_length=self.protein_maxlen,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self.device)

        with self.maybe_autocast():
            protein_feature1 = self.protein_model.esm(
                **protein1
            )  # last_hidden_states: [bs, prot_len, esm_hidden_size]
            query_protein1 = self.query_protein1.expand(batch_size, -1, -1)
            attn_mask_1 = (
                1
                - protein1.attention_mask.repeat(self.ca_num_head, 1)
                .unsqueeze(1)
                .expand(-1, self.num_query_tokens_protein1, -1)
            ).to(bool)
            p_feature1 = self.pooler_protein1(
                query_protein1,
                protein_feature1[0],
                protein_feature1[0],
                attn_mask=attn_mask_1,
            )
            protein1_embeds = self.proj_protein1(p_feature1[0])

            if protein2 is not None:
                p_feature2 = self.protein_model.esm(**protein2)
                query_protein2 = self.query_protein2.expand(batch_size, -1, -1)
                attn_mask_2 = (
                    1
                    - protein2.attention_mask.repeat(self.ca_num_head, 1)
                    .unsqueeze(1)
                    .expand(-1, self.num_query_tokens_protein2, -1)
                ).to(bool)
                delta_feature = p_feature2[0] - protein_feature1[0]
                p_feature2 = self.pooler_protein2(
                    query_protein2, delta_feature, delta_feature, attn_mask=attn_mask_2
                )
                protein2_embeds = self.proj_protein2(p_feature2[0])

        if protein2 is not None:
            return protein1_embeds, protein2_embeds
        else:
            return protein1_embeds

    def add_padding(
        self,
        wrapped_embeds,
        wrapped_attention_mask=None,
        targets=None,
        regress_ids=None,
        padding="right",
    ):
        assert (targets is None) or (regress_ids is None)
        batch_size = len(wrapped_embeds)
        max_length_batch = max([x.shape[1] for x in wrapped_embeds])
        for i in range(batch_size):
            pad_len = max_length_batch - wrapped_embeds[i].shape[1]
            if padding == "right":
                wrapped_embeds[i] = torch.cat(
                    (
                        wrapped_embeds[i],
                        torch.zeros(
                            (1, pad_len, wrapped_embeds[i].shape[2]),
                            dtype=wrapped_embeds[i].dtype,
                        ).to(wrapped_embeds[i].device),
                    ),
                    dim=1,
                )
                if wrapped_attention_mask:
                    wrapped_attention_mask[i] = torch.cat(
                        (
                            wrapped_attention_mask[i],
                            torch.zeros(
                                (1, pad_len), dtype=wrapped_attention_mask[i].dtype
                            ).to(wrapped_attention_mask[i].device),
                        ),
                        dim=1,
                    )
                if targets:
                    targets[i] = torch.cat(
                        (
                            targets[i],
                            torch.ones((1, pad_len), dtype=targets[i].dtype)
                            .to(targets[i].device)
                            .fill_(-100),
                        ),
                        dim=1,
                    )
                if regress_ids:
                    regress_ids[i] = torch.cat(
                        (
                            regress_ids[i],
                            torch.zeros((pad_len), dtype=regress_ids[i].dtype).to(
                                regress_ids[i].device
                            ),
                        ),
                        dim=0,
                    )
            else:
                wrapped_embeds[i] = torch.cat(
                    (
                        torch.zeros(
                            (1, pad_len, wrapped_embeds[i].shape[2]),
                            dtype=wrapped_embeds[i].dtype,
                        ).to(wrapped_embeds[i].device),
                        wrapped_embeds[i],
                    ),
                    dim=1,
                )
                if wrapped_attention_mask:
                    wrapped_attention_mask[i] = torch.cat(
                        (
                            torch.zeros(
                                (1, pad_len), dtype=wrapped_attention_mask[i].dtype
                            ).to(wrapped_attention_mask[i].device),
                            wrapped_attention_mask[i],
                        ),
                        dim=1,
                    )
                if targets:
                    targets[i] = torch.cat(
                        (
                            torch.ones((1, pad_len), dtype=targets[i].dtype)
                            .to(targets[i].device)
                            .fill_(-100),
                            targets[i],
                        ),
                        dim=1,
                    )
                if regress_ids:
                    regress_ids[i] = torch.cat(
                        (
                            torch.zeros((pad_len), dtype=regress_ids[i].dtype).to(
                                regress_ids[i].device
                            ),
                            regress_ids[i],
                        ),
                        dim=0,
                    )

        if targets:
            return (
                torch.cat(wrapped_embeds, dim=0),
                torch.cat(wrapped_attention_mask, dim=0),
                torch.cat(targets, dim=0),
            )
        if regress_ids:
            return (
                torch.cat(wrapped_embeds, dim=0),
                torch.cat(wrapped_attention_mask, dim=0),
                torch.stack(regress_ids, dim=0),
            )
        if wrapped_attention_mask is None:
            return torch.cat(wrapped_embeds, dim=0)
        else:
            return torch.cat(wrapped_embeds, dim=0), torch.cat(
                wrapped_attention_mask, dim=0
            )

    def _wrapped_sentence_pt(self, protein, text):
        if self.t2m:
            soft_embeds = self.soft_tokens.to(self.device)
            batched_embeds2, batched_attn_mask2, batched_soft_ids = [], [], []

        with self.maybe_autocast():
            batch_size = len(protein)
            protein = self.protein_tokenizer(
                protein,
                max_length=self.protein_maxlen,
                padding=True,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(self.device)
            p_feature = self.protein_model.esm(**protein)

            query_protein = self.query_protein1.expand(batch_size, -1, -1)
            attn_mask_ca = (
                1
                - protein.attention_mask.repeat(self.ca_num_head, 1)
                .unsqueeze(1)
                .expand(-1, self.num_query_tokens_protein1, -1)
            ).to(bool)
            pooled_feature = self.pooler_protein1(
                query_protein, p_feature[0], p_feature[0], attn_mask=attn_mask_ca
            )
            protein_embeds = self.proj_protein1(pooled_feature[0])

            input_emb = self.llm.get_input_embeddings()
            bos_tokens = (
                self.llm_tokenizer("<s>", return_tensors="pt", add_special_tokens=False)
                .to(self.device)
                .input_ids.expand(batch_size, -1)
            )
            bos_embeds = input_emb(bos_tokens)
            bop_embeds = self.bop_embeds.expand(batch_size, -1, -1)
            eop_embeds = self.eop_embeds.expand(batch_size, -1, -1)

            text = [t + "</s>" for t in text]
            text_tokens = self.llm_tokenizer(
                text,
                max_length=self.text_maxlen,
                padding=True,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            text_embeds = input_emb(text_tokens.input_ids)

            wrapped_embeds = torch.cat(
                [bos_embeds, bop_embeds, protein_embeds, eop_embeds, text_embeds], dim=1
            )
            attention_mask = torch.ones(
                (
                    batch_size,
                    bos_embeds.shape[1]
                    + bop_embeds.shape[1]
                    + protein_embeds.shape[1]
                    + eop_embeds.shape[1],
                ),
                dtype=torch.long,
                device=self.device,
            )
            wrapped_attention_mask = torch.cat(
                [attention_mask, text_tokens.attention_mask], dim=1
            )
            labels = text_tokens.input_ids.masked_fill(
                ~text_tokens.attention_mask.bool(), -100
            )
            wrapped_labels = torch.cat([attention_mask * -100, labels], dim=1)

            if self.t2m:
                for t in text:
                    tokens = self.llm_tokenizer(
                        [t.rstrip("</s>")],
                        max_length=self.text_maxlen,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(self.device)
                    text_embeds = input_emb(tokens.input_ids)
                    # regression loss
                    regress_start_id = text_embeds.shape[1] + 2
                    wrapped_embeds2 = torch.cat(
                        [
                            bos_embeds[0].unsqueeze(0),
                            text_embeds,
                            bop_embeds[0].unsqueeze(0),
                            soft_embeds,
                        ],
                        dim=1,
                    )
                    wrapped_attn_mask2 = torch.ones(
                        (1, wrapped_embeds2.shape[1]),
                        dtype=torch.long,
                        device=self.device,
                    )
                    regress_ids = torch.cat(
                        [
                            torch.zeros(
                                regress_start_id, dtype=torch.long, device=self.device
                            ),
                            torch.ones(
                                self.num_query_tokens_protein2,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    ).bool()
                    batched_embeds2.append(wrapped_embeds2)
                    batched_attn_mask2.append(wrapped_attn_mask2)
                    batched_soft_ids.append(regress_ids)
                batched_embeds2, batched_attn_mask2, batched_soft_ids = (
                    self.add_padding(
                        batched_embeds2,
                        batched_attn_mask2,
                        targets=None,
                        regress_ids=batched_soft_ids,
                    )
                )
                return (
                    wrapped_embeds,
                    wrapped_attention_mask,
                    wrapped_labels,
                    batched_embeds2,
                    batched_attn_mask2,
                    batched_soft_ids,
                )

        return wrapped_embeds, wrapped_attention_mask, wrapped_labels

    def _wrapped_sentence_ft(
        self, protein1_embeds, protein2_embeds, mut_entry, p_function, muta_prompt, text
    ):
        assert text is not None
        batch_size = protein1_embeds.shape[0]
        input_emb = self.llm.get_input_embeddings()
        bos_tokens = (
            self.llm_tokenizer("<s>", return_tensors="pt", add_special_tokens=False)
            .to(self.device)
            .input_ids
        )
        bos_embeds = input_emb(bos_tokens)  # [1, 1, 4096]
        bop_embeds = self.bop_embeds.to(self.device)
        eop_embeds = self.eop_embeds.to(self.device)
        bom_embeds = self.bom_embeds.to(self.device)
        eom_embeds = self.eom_embeds.to(self.device)

        if self.t2m:
            soft_embeds = self.soft_tokens.to(self.device)
            batched_embeds2, batched_attn_mask2, batched_regress_ids = [], [], []

        batched_embeds1, batched_attn_mask1, batched_labels = [], [], []
        p_function = [t + "</s>" for t in p_function]
        text = [t + "</s>" for t in text]
        sys_prompt_tokens = (
            self.llm_tokenizer(
                "You are an expert at biology and life science. Now a user gives you several protein sequences and mutations. Please follow user instructions and answer their questions. Based on the following protein sequence, please describe its function.",
                max_length=self.func_maxlen,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            .to(self.device)
            .input_ids
        )
        sys_embeds = input_emb(sys_prompt_tokens)
        for i in range(batch_size):
            function_tokens = self.llm_tokenizer(
                p_function[i],
                max_length=self.func_maxlen,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            mutation_tokens = self.llm_tokenizer(
                muta_prompt[i],
                max_length=self.text_maxlen,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            text_tokens = self.llm_tokenizer(
                text[i],
                max_length=self.text_maxlen,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.device)
            func_embeds = input_emb(function_tokens.input_ids)
            muta_embeds = input_emb(mutation_tokens.input_ids)
            text_embeds = input_emb(text_tokens.input_ids)

            # understanding loss
            wrapped_embeds1 = torch.cat(
                [
                    bos_embeds,
                    sys_embeds,
                    bop_embeds,
                    protein1_embeds[i].unsqueeze(0),
                    eop_embeds,
                    func_embeds,
                    muta_embeds,
                    bom_embeds,
                    protein2_embeds[i].unsqueeze(0),
                    eom_embeds,
                    text_embeds,
                ],
                dim=1,
            )
            wrapped_attn_mask1 = torch.ones(
                (1, wrapped_embeds1.shape[1]), dtype=torch.long, device=self.device
            )
            wrapped_labels = torch.cat(
                [
                    torch.ones(
                        (1, 3 + sys_embeds.shape[1] + protein2_embeds.shape[1]),
                        dtype=torch.long,
                        device=self.device,
                    )
                    * -100,
                    function_tokens.input_ids,
                    torch.ones(
                        (1, muta_embeds.shape[1] + 2 + protein2_embeds.shape[1]),
                        dtype=torch.long,
                        device=self.device,
                    )
                    * -100,
                    text_tokens.input_ids,
                ],
                dim=1,
            )
            batched_embeds1.append(wrapped_embeds1)
            batched_attn_mask1.append(wrapped_attn_mask1)
            batched_labels.append(wrapped_labels)

            if self.t2m:
                regress_start_id = (
                    sys_embeds.shape[1]
                    + self.num_query_tokens_protein1
                    + 3
                    + func_embeds.shape[1]
                    + text_embeds.shape[1]
                )
                wrapped_embeds2 = torch.cat(
                    [
                        bos_embeds,
                        sys_embeds,
                        bop_embeds,
                        protein1_embeds[i].unsqueeze(0),
                        eop_embeds,
                        func_embeds,
                        text_embeds[:, :-1, :],
                        bom_embeds,
                        soft_embeds,
                        eom_embeds,
                        text_embeds[:, -1:, :],
                    ],
                    dim=1,
                )
                wrapped_attn_mask2 = torch.ones(
                    (1, wrapped_embeds2.shape[1]), dtype=torch.long, device=self.device
                )
                regress_ids = torch.cat(
                    [
                        torch.zeros(
                            regress_start_id, dtype=torch.long, device=self.device
                        ),
                        torch.ones(
                            self.num_query_tokens_protein2,
                            dtype=torch.long,
                            device=self.device,
                        ),
                        torch.zeros(2, dtype=torch.long, device=self.device),
                    ],
                    dim=0,
                ).bool()
                batched_embeds2.append(wrapped_embeds2)
                batched_attn_mask2.append(wrapped_attn_mask2)
                batched_regress_ids.append(regress_ids)

        batched_embeds1, batched_attn_mask1, batched_labels = self.add_padding(
            batched_embeds1,
            batched_attn_mask1,
            targets=batched_labels,
            regress_ids=None,
        )
        if self.t2m:
            mut_pos = torch.tensor(
                [int(x[1:-1]) for x in mut_entry], dtype=torch.long
            ).to(self.device)
            mut_aa = self.protein_tokenizer(
                [x[-1] for x in mut_entry],
                padding=False,
                truncation=True,
                max_length=self.protein_maxlen,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device)
            batched_embeds2, batched_attn_mask2, batched_regress_ids = self.add_padding(
                batched_embeds2,
                batched_attn_mask2,
                targets=None,
                regress_ids=batched_regress_ids,
            )
            return (
                batched_embeds1,
                batched_attn_mask1,
                batched_labels,
                batched_embeds2,
                batched_attn_mask2,
                batched_regress_ids,
                mut_pos,
                mut_aa,
            )
        else:
            return batched_embeds1, batched_attn_mask1, batched_labels

    def _wrapped_sentence_inference(
        self,
        protein1_embeds,
        protein2_embeds,
        muta_prompt,
        predict_function=None,
        mut_text=None,
    ):
        batch_size = protein1_embeds.shape[0]
        input_emb = self.llm.get_input_embeddings()
        bos_tokens = (
            self.llm_tokenizer("<s>", return_tensors="pt", add_special_tokens=False)
            .to(self.device)
            .input_ids
        )
        bos_embeds = input_emb(bos_tokens)  # [1, 1, 4096]
        sys_prompt_tokens = (
            self.llm_tokenizer(
                "You are an expert at biology and life science. Now a user gives you several protein sequences and mutations. Please follow user instructions and answer their questions.",
                max_length=self.func_maxlen,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            .to(self.device)
            .input_ids
        )
        sys_embeds = input_emb(sys_prompt_tokens)
        if predict_function is None:  # CoT stage 1
            sys_embeds = sys_embeds.expand(batch_size, -1, -1)
            bos_embeds = bos_embeds.expand(batch_size, -1, -1)
            bop_embeds = self.bop_embeds.expand(batch_size, -1, -1)
            eop_embeds = self.eop_embeds.expand(batch_size, -1, -1)
            bom_embeds = self.bom_embeds.expand(batch_size, -1, -1)
            eom_embeds = self.eom_embeds.expand(batch_size, -1, -1)
            wrapped_embeds = torch.cat(
                [bos_embeds, sys_embeds, bop_embeds, protein1_embeds, eop_embeds], dim=1
            )
            attention_mask = torch.ones(
                (batch_size, wrapped_embeds.shape[1]),
                dtype=torch.long,
                device=self.device,
            )
            return wrapped_embeds, attention_mask

        else:  # CoT stage 2
            bop_embeds = self.bop_embeds.to(self.device)
            eop_embeds = self.eop_embeds.to(self.device)
            bom_embeds = self.bom_embeds.to(self.device)
            eom_embeds = self.eom_embeds.to(self.device)
            batched_embeds, batched_attn_mask = [], []
            if mut_text is not None:
                batched_regress_ids = []
            predict_function = [t + "</s>" for t in predict_function]
            for i in range(batch_size):
                function_tokens = self.llm_tokenizer(
                    predict_function[i],
                    max_length=self.func_maxlen,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)
                mutation_tokens = self.llm_tokenizer(
                    muta_prompt[i],
                    max_length=self.text_maxlen,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).to(self.device)
                func_embeds = input_emb(function_tokens.input_ids)
                muta_embeds = input_emb(mutation_tokens.input_ids)
                if mut_text is not None:
                    mut_eff = self.llm_tokenizer(
                        mut_text[i],
                        max_length=self.text_maxlen,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(self.device)
                    mut_eff_embeds = input_emb(mut_eff.input_ids)
                    soft_embeds = self.soft_tokens.to(self.device)
                    regress_start_id = (
                        sys_embeds.shape[1]
                        + self.num_query_tokens_protein1
                        + 4
                        + func_embeds.shape[1]
                        + mut_eff_embeds.shape[1]
                    )
                    wrapped_embeds = torch.cat(
                        [
                            bos_embeds,
                            sys_embeds,
                            bop_embeds,
                            protein1_embeds[i].unsqueeze(0),
                            eop_embeds,
                            func_embeds,
                            mut_eff_embeds,
                            bom_embeds,
                            soft_embeds,
                        ],
                        dim=1,
                    )
                    regress_ids = torch.cat(
                        [
                            torch.zeros(
                                regress_start_id, dtype=torch.long, device=self.device
                            ),
                            torch.ones(
                                self.num_query_tokens_protein2,
                                dtype=torch.long,
                                device=self.device,
                            ),
                        ],
                        dim=0,
                    ).bool()
                    batched_regress_ids.append(regress_ids)
                else:
                    wrapped_embeds = torch.cat(
                        [
                            bos_embeds,
                            sys_embeds,
                            bop_embeds,
                            protein1_embeds[i].unsqueeze(0),
                            eop_embeds,
                            func_embeds,
                            muta_embeds,
                            bom_embeds,
                            protein2_embeds[i].unsqueeze(0),
                            eom_embeds,
                        ],
                        dim=1,
                    )
                wrapped_attn_mask = torch.ones(
                    (1, wrapped_embeds.shape[1]), dtype=torch.long, device=self.device
                )
                batched_embeds.append(wrapped_embeds)
                batched_attn_mask.append(wrapped_attn_mask)

            if mut_text is None:
                batched_embeds, batched_attn_mask = self.add_padding(
                    batched_embeds,
                    batched_attn_mask,
                    targets=None,
                    regress_ids=None,
                    padding="left",
                )
                return batched_embeds, batched_attn_mask
            else:
                batched_embeds, batched_attn_mask, batched_regress_ids = (
                    self.add_padding(
                        batched_embeds,
                        batched_attn_mask,
                        targets=None,
                        regress_ids=batched_regress_ids,
                        padding="left",
                    )
                )
                return batched_embeds, batched_attn_mask, batched_regress_ids

    def protein_mask(self, protein, mask_ratio=0.15):
        protein = self.protein_tokenizer(
            protein,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            max_length=self.protein_maxlen,
            return_tensors="pt",
        ).to(self.device)
        labels = protein.input_ids.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape, mask_ratio)).bool()
        masked_indices[labels == self.protein_tokenizer.pad_token_id] = False
        masked_indices[labels == self.protein_tokenizer.cls_token_id] = False
        masked_indices[labels == self.protein_tokenizer.eos_token_id] = False
        protein.input_ids[masked_indices] = self.protein_tokenizer.mask_token_id
        labels[~masked_indices] = -100
        return protein, labels

    def forward_pt(self, protein, text):
        if self.t2m:
            (
                input_embeds_p2t,
                attn_mask_p2t,
                labels_p2t,
                input_embeds_t2p,
                attn_mask_t2p,
                soft_ids,
            ) = self._wrapped_sentence_pt(protein, text)
        else:
            input_embeds_p2t, attn_mask_p2t, labels_p2t = self._wrapped_sentence_pt(
                protein, text
            )
        with self.maybe_autocast():
            if self.m2t:
                loss_p2t = self.llm(
                    inputs_embeds=input_embeds_p2t,
                    attention_mask=attn_mask_p2t,
                    labels=labels_p2t,
                    return_dict=True,
                ).loss
            if self.t2m:
                masked_protein, masked_labels = self.protein_mask(protein)
                outputs = self.llm(
                    inputs_embeds=input_embeds_t2p,
                    attention_mask=attn_mask_t2p,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states[-1]
                soft_embeds = outputs[soft_ids].contiguous()
                soft_embeds = self.proj_text(
                    soft_embeds.view(len(protein), self.num_query_tokens_protein2, -1)
                )
                loss_t2p = torch.mean(
                    self.protein_model(
                        input_ids=masked_protein.input_ids,
                        attention_mask=masked_protein.attention_mask,
                        encoder_hidden_states=soft_embeds,
                        encoder_attention_mask=torch.ones(
                            soft_embeds.shape[:-1], dtype=torch.long
                        ).to(self.device),
                        labels=masked_labels,
                        return_dict=True,
                    ).loss
                )

            if self.m2t and self.t2m:
                return loss_p2t + loss_t2p, {"loss_p2t": loss_p2t, "loss_t2p": loss_t2p}
            elif self.m2t:
                return loss_p2t, {"loss_p2t": loss_p2t}
            else:
                return loss_t2p, {"loss_t2p": loss_t2p}

    def forward_ft(self, protein1, protein2, mut_entry, text, p_function, muta_prompt):
        protein1_embeds, protein2_embeds = self._encode_protein(protein1, protein2)
        if self.t2m:
            (
                input_embeds_m2t,
                attn_mask_m2t,
                labels_m2t,
                input_embeds_t2m,
                attn_mask_t2m,
                soft_ids_t2m,
                mut_pos,
                mut_aa,
            ) = self._wrapped_sentence_ft(
                protein1_embeds,
                protein2_embeds,
                mut_entry,
                p_function,
                muta_prompt,
                text,
            )
        else:
            input_embeds_m2t, attn_mask_m2t, labels_m2t = self._wrapped_sentence_ft(
                protein1_embeds,
                protein2_embeds,
                mut_entry,
                p_function,
                muta_prompt,
                text,
            )

        with self.maybe_autocast():
            if self.m2t:
                loss_m2t = self.llm(
                    inputs_embeds=input_embeds_m2t,
                    attention_mask=attn_mask_m2t,
                    labels=labels_m2t,
                    return_dict=True,
                ).loss

            if self.t2m:
                outputs = self.llm(
                    inputs_embeds=input_embeds_t2m,
                    attention_mask=attn_mask_t2m,
                    output_hidden_states=True,
                    return_dict=True,
                ).hidden_states
                soft_output = outputs[soft_ids_t2m].contiguous()
                soft_output = self.proj_text(
                    soft_output.view(len(protein1), self.num_query_tokens_protein2, -1)
                )
                protein = self.protein_tokenizer(
                    protein1,
                    add_special_tokens=True,
                    truncation=True,
                    padding=True,
                    max_length=self.protein_maxlen,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.protein_model(
                    input_ids=protein.input_ids,
                    attention_mask=protein.attention_mask,
                    mutation_position=mut_pos,
                    mutation_aa=mut_aa,
                    batch_idx=torch.arange(len(protein1)).to(self.device),
                    encoder_hidden_states=soft_output,
                    encoder_attention_mask=torch.ones(
                        soft_output.shape[:1], dtype=torch.long
                    ).to(self.device),
                    return_dict=True,
                )

            if self.m2t and self.t2m:
                return loss_m2t + outputs.loss_pos + 0.2 * outputs.loss_aa, {
                    "loss_m2t": loss_m2t,
                    "loss_pos": outputs.loss_pos,
                    "loss_aa": outputs.loss_aa,
                }
            elif self.m2t:
                return loss_m2t, {"loss_m2t": loss_m2t}
            else:
                return outputs.loss_pos + 0.2 * outputs.loss_aa, {
                    "loss_pos": outputs.loss_pos,
                    "loss_aa": outputs.loss_aa,
                }

    @torch.no_grad()
    def generate(
        self,
        protein1,
        protein2,
        muta_prompt,
        pfunction=None,
        use_gt_function=False,
        use_nucleus_sampling=True,
        num_beams=2,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            # stage 1
            protein1_embeds, protein2_embeds = self._encode_protein(protein1, protein2)
            if not use_gt_function:
                input_embeds, attn_mask = self._wrapped_sentence_inference(
                    protein1_embeds, protein2_embeds, muta_prompt, predict_function=None
                )
                outputs_function = self.llm.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attn_mask,
                    do_sample=use_nucleus_sampling,
                    top_p=top_p,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_length=max_length,
                    min_length=min_length,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    pad_token_id=self.llm_tokenizer.pad_token_id,
                )
                outputs_function[outputs_function == 0] = (
                    2  # convert output id 0 to 2 (eos_token_id)
                )
                output_function_text = self.llm_tokenizer.batch_decode(
                    outputs_function, skip_special_tokens=True
                )
                output_function_text = [text.strip() for text in output_function_text]
            else:  # use ground truth protein function directly
                output_function_text = pfunction

            # stage 2
            input_embeds, attn_mask = self._wrapped_sentence_inference(
                protein1_embeds,
                protein2_embeds,
                muta_prompt,
                predict_function=output_function_text,
            )
            outputs_effect = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )
            outputs_effect[outputs_effect == 0] = (
                2  # convert output id 0 to 2 (eos_token_id)
            )
            output_effect_text = self.llm_tokenizer.batch_decode(
                outputs_effect, skip_special_tokens=True
            )
            output_effect_text = [text.strip() for text in output_effect_text]

        return output_function_text, output_effect_text

    @torch.no_grad()
    def lm_design(
        self,
        protein,
        text,
        muta_prompt=None,
        pfunction=None,
        use_gt_function=True,
        use_nucleus_sampling=True,
        num_beams=2,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        protein_embeds = self._encode_protein(protein, None)
        if not use_gt_function:
            input_embeds, attn_mask = self._wrapped_sentence_inference(
                protein_embeds, None, None, predict_function=None
            )
            outputs_function = self.llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                pad_token_id=self.llm_tokenizer.pad_token_id,
            )
            outputs_function[outputs_function == 0] = (
                2  # convert output id 0 to 2 (eos_token_id)
            )
            output_function_text = self.llm_tokenizer.batch_decode(
                outputs_function, skip_special_tokens=True
            )
            output_function_text = [text.strip() for text in output_function_text]
        else:
            output_function_text = pfunction
        input_embeds, attn_mask, soft_ids = self._wrapped_sentence_inference(
            protein_embeds,
            None,
            muta_prompt,
            predict_function=output_function_text,
            mut_text=text,
        )
        soft_output = self.llm.model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        soft_output = soft_output[soft_ids].contiguous()
        soft_output = self.proj_text(
            soft_output.view(len(protein), self.num_query_tokens_protein2, -1)
        )
        protein = self.protein_tokenizer(
            protein,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.protein_maxlen,
            return_tensors="pt",
        ).to(self.device)
        return self.protein_model.lm_design(
            input_ids=protein.input_ids,
            attention_mask=protein.attention_mask,
            encoder_hidden_states=soft_output,
            encoder_attention_mask=torch.ones(
                soft_output.shape[:-1], dtype=torch.long
            ).to(self.device),
        )
