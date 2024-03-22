import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--n_hidden", type=int, nargs="+", default=[32, 64, 64, 32])
    parser.add_argument("--n_T", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--deg_type", type=str, default="blur")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--beta1", type=float, default=1e-4)
    parser.add_argument("--beta2", type=float, default=0.02)
    parser.add_argument("--net", type=str, default="UNet")
    parser.add_argument("--use_ckpt", type=int, default=0)
    parser.add_argument("--in_channels", type=int, default=1)
    name = name_run(parser)
    parser.add_argument("--name", type=str, default=name)

    opt = parser.parse_known_args()[0]

    return opt, str_options(parser, opt)


def name_run(parser):
    """Names run based on non-default options."""
    opt = parser.parse_known_args()[0]
    non_default = {k: v for k, v in vars(opt).items() if v != parser.get_default(k)}

    for key in ["n_epoch", "in_channels", "use_ckpt"]:
        non_default.pop(key, None)

    if non_default == {}:
        name = "default"

    else:
        non_default = dict(sorted(non_default.items()))
        name = "_".join(
            [
                f"{k}={v}" if not isinstance(v, str) else v
                for k, v in non_default.items()
            ]
        )
        name = name.replace(" ", "")
        if opt.in_channels == 3:
            name += "_RGB"
    return name


def str_options(parser, opt):
    """Convert opt to string."""
    message = ""
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    return message
