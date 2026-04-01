# ============================================================
# Customer Churn Prediction
# ============================================================

source("config.R")

library(ggplot2)
library(caret)
library(corrplot)
library(car)
library(dplyr)
library(tidyr)
library(pROC)
library(randomForest)
options(warn = -1)

# ---- Load Data ----

df.train <- read.csv(TRAINING_DATA_PATH)
df.test  <- read.csv(TESTING_DATA_PATH)

str(df.train)

# ---- Preprocessing ----

preprocess <- function(df) {
  df$CustomerID         <- NULL
  df$Gender             <- as.factor(df$Gender)
  df$Subscription.Type  <- as.factor(df$Subscription.Type)
  df$Contract.Length    <- as.factor(df$Contract.Length)
  df$Churn              <- as.factor(df$Churn)
  return(df)
}

df.train <- preprocess(df.train)
df.train$Churn.num <- as.numeric(as.character(df.train$Churn))

df.test <- preprocess(df.test)

summary(df.train)
summary(df.test)

# ---- Missing Value Check ----

check_missing <- function(df, df.name = "") {
  missing.idx <- which(!complete.cases(df))
  cat(df.name, ": Found", length(missing.idx), "missing rows\n")
  if (length(missing.idx) > 0) print(df[missing.idx, ], row.names = FALSE)
  invisible(missing.idx)
}

check_missing(df.train, "df.train")
check_missing(df.test, "df.test")

df.train <- df.train[complete.cases(df.train), ]

# ---- Feature Distribution ----

ggplot(df.train, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Churn Distribution (Training Set)",
       x = "", y = "Count") +
  theme(legend.position = "none")

# ---- Categorical Features ----

categorical.features <- c("Gender", "Subscription.Type", "Contract.Length")

df.cat.long <- pivot_longer(data = df.train[, c("Churn", categorical.features)],
                            cols = -Churn,
                            names_to = "feature",
                            values_to = "value")

ggplot(df.cat.long, aes(x = value, fill = Churn)) +
  geom_bar(position = "fill") +
  facet_wrap(~feature, scales = "free_x") +
  labs(title = "Feature Distribution by Churn: Categorical Features",
       x = "", y = "Proportion")

# ---- Numeric Features ----

numeric.features <- c("Age", "Tenure", "Usage.Frequency", "Support.Calls",
                      "Payment.Delay", "Total.Spend", "Last.Interaction")

df.long <- pivot_longer(data = df.train[, c("Churn", numeric.features)],
                        cols = -Churn,
                        names_to = "feature",
                        values_to = "value")

ggplot(df.long, aes(x = Churn, y = value, fill = Churn)) +
  geom_boxplot(outlier.size = 0.5, alpha = 0.7) +
  facet_wrap(~feature, scales = "free_y", ncol = 2) +
  labs(title = "Feature Distribution by Churn: Numeric Features",
       x = "", y = "Value") +
  theme(legend.position = "none")

# ---- Feature Correlation ----

cor.data        <- df.train[, numeric.features]
cor.data$Churn  <- df.train$Churn.num

corrplot(cor(cor.data, use = "complete.obs"),
         method = "color",
         type = "upper",
         tl.cex = 0.8,
         addCoef.col = "black",
         number.cex = 0.6)

# ---- Multicollinearity Check (VIF) ----

compute_vif <- function(data) {
  lm.proxy   <- lm(Churn.num ~ . - Churn, data = data)
  vif.result <- vif(lm.proxy)
  vif.data   <- data.frame(
    Feature = rownames(vif.result),
    VIF     = round(vif.result[, 1], 3)
  )
  print(vif.data, row.names = FALSE)
  invisible(vif.data)
}

compute_vif(df.train)

# ---- Feature Selection ----

feature_select <- function(data, k = 2) {
  full.glm    <- glm(Churn ~ . - Churn.num, data = data, family = "binomial")
  step.result <- stats::step(full.glm, direction = "backward", k = k)
  summary(step.result)
  invisible(step.result)
}

aic.result <- feature_select(df.train)

# ---- Model Training ----

ctrl <- trainControl(method = "cv", number = 5)

train_glm <- function(data, ctrl) {
  set.seed(42)
  model <- train(Churn ~ Age + Gender + Tenure + Usage.Frequency +
                   Support.Calls + Payment.Delay + Subscription.Type + Contract.Length +
                   Total.Spend + Last.Interaction,
                 data      = data,
                 method    = "glm",
                 family    = "binomial",
                 trControl = ctrl,
                 maxit     = 300,
                 trace     = FALSE)
  return(model)
}

train_glm_bic <- function(data, ctrl) {
  set.seed(42)
  model <- train(Churn ~ Age + Gender + Usage.Frequency +
                   Support.Calls + Payment.Delay + Subscription.Type + Contract.Length +
                   Total.Spend + Last.Interaction,
                 data      = data,
                 method    = "glm",
                 family    = "binomial",
                 trControl = ctrl,
                 maxit     = 300,
                 trace     = FALSE)
  return(model)
}

train_rf <- function(data, ctrl, sample.size = 50000) {
  set.seed(42)
  idx   <- sample(nrow(data), sample.size)
  model <- train(Churn ~ Age + Gender + Tenure + Usage.Frequency +
                   Support.Calls + Payment.Delay + Subscription.Type +
                   Contract.Length + Total.Spend + Last.Interaction,
                 data      = data[idx, ],
                 method    = "rf",
                 trControl = ctrl)
  return(model)
}

eval_model <- function(model, test.data) {
  pred    <- predict(model, newdata = test.data, type = "raw")
  prob    <- predict(model, newdata = test.data, type = "prob")[, "1"]
  cm      <- confusionMatrix(pred, test.data$Churn, positive = "1")
  auc_val <- auc(roc(as.numeric(as.character(test.data$Churn)), prob, quiet = TRUE))

  print(cm$table)

  results <- data.frame(
    Accuracy    = round(cm$overall["Accuracy"], 4),
    Sensitivity = round(cm$byClass["Sensitivity"], 4),
    Specificity = round(cm$byClass["Specificity"], 4),
    F1          = round(cm$byClass["F1"], 4),
    AUC         = round(auc_val, 4)
  )
  print(results, row.names = FALSE)
  invisible(results)
}

# ---- Original Split ----

cv.model.original <- train_glm(df.train, ctrl)
eval_model(cv.model.original, df.test)

# ---- Distribution Shift Analysis ----

ggplot() +
  geom_density(data = df.train[df.train$Churn == 0, ],
               aes(x = Support.Calls, fill = "Train (Not Churn)"), alpha = 0.5) +
  geom_density(data = df.train[df.train$Churn == 1, ],
               aes(x = Support.Calls, fill = "Train (Churn)"), alpha = 0.5) +
  geom_density(data = df.test[df.test$Churn == 0, ],
               aes(x = Support.Calls, fill = "Test (Not Churn)"), alpha = 0.5) +
  labs(title = "Support.Calls Distribution: Train vs Test",
       x = "", y = "Density", fill = "Dataset")

ggplot() +
  geom_density(data = df.train[df.train$Churn == 0, ],
               aes(x = Total.Spend, fill = "Train (Not Churn)"), alpha = 0.5) +
  geom_density(data = df.train[df.train$Churn == 1, ],
               aes(x = Total.Spend, fill = "Train (Churn)"), alpha = 0.5) +
  geom_density(data = df.test[df.test$Churn == 0, ],
               aes(x = Total.Spend, fill = "Test (Not Churn)"), alpha = 0.5) +
  labs(title = "Total.Spend Distribution: Train vs Test",
       x = "", y = "Density", fill = "Dataset")

# ---- Re-split ----

df.all <- rbind(df.train[, names(df.test)], df.test)

set.seed(42)
train_idx    <- sample(nrow(df.all), 0.8 * nrow(df.all))
df.train.new <- droplevels(preprocess(df.all[train_idx, ]))
df.test.new  <- droplevels(preprocess(df.all[-train_idx, ]))
df.train.new$Churn.num <- as.numeric(as.character(df.train.new$Churn))

# ---- Feature Selection (Re-split) ----

aic.result <- feature_select(df.train.new, k = 2)
bic.result <- feature_select(df.train.new, k = log(nrow(df.train.new)))

formula(aic.result)
formula(bic.result)

# ---- Re-split: Logistic Regression ----

cv.model.aic <- train_glm(df.train.new, ctrl)
eval_model(cv.model.aic, df.test.new)

cv.model.bic <- train_glm_bic(df.train.new, ctrl)
eval_model(cv.model.bic, df.test.new)

# ---- Re-split: Random Forest ----

rf.model <- train_rf(df.train.new, ctrl)
eval_model(rf.model, df.test.new)

plot(varImp(rf.model))

# ---- Coefficient Analysis ----

summary(cv.model.aic$finalModel)

coef.table <- summary(cv.model.aic$finalModel)$coefficients
odds.ratio  <- exp(coef.table[, 1])
ci          <- exp(confint.default(cv.model.aic$finalModel))

result <- data.frame(
  Feature     = rownames(coef.table),
  Coefficient = round(coef.table[, 1], 3),
  Odds.Ratio  = round(odds.ratio, 3),
  CI.Lower    = round(ci[, 1], 3),
  CI.Upper    = round(ci[, 2], 3),
  P.Value     = format(coef.table[, 4], digits = 3)
)
print(result, row.names = FALSE)

# ---- Overall Comparison ----

res.original <- eval_model(cv.model.original, df.test)
res.aic      <- eval_model(cv.model.aic, df.test.new)
res.bic      <- eval_model(cv.model.bic, df.test.new)
res.rf       <- eval_model(rf.model, df.test.new)

all.results        <- rbind(res.original, res.aic, res.bic, res.rf)
all.results$Model  <- c("LR Original", "LR AIC (Re-split)", "LR BIC (Re-split)", "RF (Re-split)")
all.results        <- all.results[, c("Model", "Accuracy", "Sensitivity", "Specificity", "F1", "AUC")]

print(all.results, row.names = FALSE)