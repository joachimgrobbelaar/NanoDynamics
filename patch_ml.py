import re

with open("ai/train_transition.py", "r") as f:
    content = f.read()

# Fix loading
load_target = """                model.load_state_dict(prior["state_dict"])
                print(f"Resumed model weights from {ckpt_path}")"""
load_replacement = """                model.load_state_dict(prior["state_dict"])
                if "optimizer_state" in prior:
                    opt.load_state_dict(prior["optimizer_state"])
                print(f"Resumed model and optimizer state from {ckpt_path}")"""
content = content.replace(load_target, load_replacement)

# Fix saving in the loop
save_target = """                ckpt = {"state_dict": model.state_dict(), "stats": stats,
                        "config": {"hidden_dim": hidden, "hidden_layers": layers},
                        "epoch": epoch, "loss": current_loss}"""
save_replacement = """                ckpt = {"state_dict": model.state_dict(), "optimizer_state": opt.state_dict(), "stats": stats,
                        "config": {"hidden_dim": hidden, "hidden_layers": layers},
                        "epoch": epoch, "loss": current_loss}"""
content = content.replace(save_target, save_replacement)

# Fix saving at the end
save_target2 = """    ckpt = {"state_dict": model.state_dict(), "stats": stats,
            "config": {"hidden_dim": hidden, "hidden_layers": layers},
            "total_epochs": epochs, "final_loss": best_loss}"""
save_replacement2 = """    ckpt = {"state_dict": model.state_dict(), "optimizer_state": opt.state_dict(), "stats": stats,
            "config": {"hidden_dim": hidden, "hidden_layers": layers},
            "total_epochs": epochs, "final_loss": best_loss}"""
content = content.replace(save_target2, save_replacement2)

with open("ai/train_transition.py", "w") as f:
    f.write(content)
