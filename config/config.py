class CONFIG:
    _instance = None
    USER_DATA_PATH = "../Course_Scraper/assets/augumented_data/augmented_user_rating.csv"

    seed = 67

    # data <-- didn't implement this little guy
    genre = False

    # bsl
    bsl_n_epochs = 20
    bsl_lr = 0.001

    # nn
    nn_epochs = 1500
    nn_lr = 0.001
    nn_patience = 100
    nn_user_emb = 32
    nn_title_emb = 32
    nn_batches = 8192

    # watchdog
    watchit = False
    watchit_model = "knn"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CONFIG, cls).__new__(cls)
        return cls._instance


cfg = CONFIG()
