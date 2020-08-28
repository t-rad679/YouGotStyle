from const import (TOTAL_VARIATION_WEIGHT, EPSILON, BETA_1, LEARNING_RATE,
                   LOG_DIR, STYLE_WEIGHT, CONTENT_WEIGHT, BETA_2)


class Config:
    def __init__(self,
                 total_variation_weight=TOTAL_VARIATION_WEIGHT,
                 epsilon=EPSILON,
                 beta_1=BETA_1,
                 beta_2=BETA_2,
                 learning_rate=LEARNING_RATE,
                 content_weight=CONTENT_WEIGHT,
                 style_weight=STYLE_WEIGHT,
                 log_dir=LOG_DIR):
        self.total_variation_weight = total_variation_weight
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.log_dir = log_dir
