import chimera
import fire

def main(train_data, max_len=100000):
    # Load the training data
    tokenizer = chimera.data.tokenizer.Tokenizer(model_max_length=max_len)

    fq_data_module  = chimera.data.fq.DataModule(tokenizer, train_data)
    fq_data_module.prepare_data()
    fq_data_module.setup()
    train_data_loader = fq_data_module.train_dataloader()
    data_iterator = iter(train_data_loader)

    print("Data loaded successfully")



if __name__ == "__main__":
    fire.Fire(main)