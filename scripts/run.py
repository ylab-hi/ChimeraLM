import chimera
import sys

def main():
    train_data_path = sys.argv[1]
    max_len = 100000

    # Load the training data
    tokenizer = chimera.data.tokenizer.Tokenizer(model_max_length=max_len)

    fq_data_module  = chimera.data.fq.DataModule(tokenizer, train_data_path)
    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()

    data_iterator = iter(train_data_loader)

    import ipdb; ipdb.set_trace()

    print("Data loaded successfully")



if __name__ == "__main__":
    main()