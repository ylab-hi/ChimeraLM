import fire

import chimera


def load_predict(path):
    return chimera.load_predicts_from_batch_pts(path)


def load_sv(path):
    res = {}
    with open(path) as f:
        for line in f:
            read, sv = line.strip().split("\t")
            res[read] = sv
    return res


def main(predict_path, sv_path):
    predicts = load_predict(predict_path)
    sv = load_sv(sv_path)

    with open("predict_with_sv.text", "w") as f:
        for read, predict in predicts.items():
            if read in sv:
                f.write(f"{read}\t{predict}\t{sv[read]}\n")
            else:
                f.write(f"{read}\t{predict}\tNA\n")


if __name__ == "__main__":
    fire.Fire(main)
