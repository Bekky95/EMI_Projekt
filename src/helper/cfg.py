CFG = {
    "data_root"      : "/kaggle/input",
    "out_dir"        : "/kaggle/working",
    "clip_model"     : "openai/clip-vit-base-patch32",      # <- ImageModel("openai/clip-vit-base-patch32")
    "mini_la_model"  : "all-MiniLM-L6-v2",  # <- TextModel("all-MiniLM-L6-v2")
    "batch_size"     : 16,
    "epochs"         : 15,
    "lr_roberta"     : 2e-5,
    "lr_head"        : 5e-4,      # ← increased
    "weight_decay"   : 1e-2,
    "max_grad_norm"  : 1.0,
    "warmup_ratio"   : 0.1,
    "patience"       : 4,
    "max_text_len"   : 128,
    "img_size"       : 224,
    "task_weights"   : {"sentiment": 1.0, "humour": 0.8, "sarcasm": 0.8, "offensive": 0.8},
    "focal_gamma"    : 2.0,
    "memory_size"    : 32,
    "fusion_dim"     : 256,
}