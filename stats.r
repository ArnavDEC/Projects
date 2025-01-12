# Load necessary libraries
library(ggplot2)
library(dplyr)

# Generate random data for analysis
set.seed(42)  # For reproducibility
n <- 200
data <- data.frame(
  ID = 1:n,
  Age = sample(20:60, n, replace = TRUE),
  Gender = sample(c("Male", "Female"), n, replace = TRUE),
  Income = rnorm(n, mean = 50000, sd = 15000),
  SpendingScore = rnorm(n, mean = 60, sd = 20)
)

# Add a categorical variable based on spending score
data$Category <- cut(data$SpendingScore,
                     breaks = c(-Inf, 40, 70, Inf),
                     labels = c("Low", "Medium", "High"))

# Summary of the data
print("Summary of the dataset:")
print(summary(data))

# Visualize Income distribution
ggplot(data, aes(x = Income)) +
  geom_histogram(binwidth = 5000, fill = "blue", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Income Distribution", x = "Income", y = "Frequency")

# Visualize SpendingScore by Gender
ggplot(data, aes(x = Gender, y = SpendingScore, fill = Gender)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Spending Score by Gender", x = "Gender", y = "Spending Score")

# Calculate correlation between Income and Spending Score
correlation <- cor(data$Income, data$SpendingScore)
cat("Correlation between Income and Spending Score:", correlation, "\n")

# Perform a t-test on Spending Score by Gender
t_test_result <- t.test(SpendingScore ~ Gender, data = data)
print("T-test results:")
print(t_test_result)

# Create a scatterplot of Income vs. SpendingScore
ggplot(data, aes(x = Income, y = SpendingScore)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", col = "red") +
  theme_minimal() +
  labs(title = "Income vs. Spending Score", x = "Income", y = "Spending Score")

# Fit a linear regression model
model <- lm(SpendingScore ~ Income + Age + Gender, data = data)
summary_model <- summary(model)
cat("Summary of the Linear Model:\n")
print(summary_model)

# Add predictions to the data
data$PredictedSpending <- predict(model, newdata = data)

# Visualize Actual vs. Predicted Spending Scores
ggplot(data, aes(x = SpendingScore, y = PredictedSpending)) +
  geom_point(alpha = 0.6, color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, col = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "Actual vs Predicted Spending Score", x = "Actual", y = "Predicted")

# Group data by Category and calculate mean Income and SpendingScore
category_summary <- data %>%
  group_by(Category) %>%
  summarize(
    MeanIncome = mean(Income, na.rm = TRUE),
    MeanSpendingScore = mean(SpendingScore, na.rm = TRUE),
    Count = n()
  )

print("Summary statistics by Category:")
print(category_summary)

# Visualize category-wise mean Income and Spending Score
ggplot(category_summary, aes(x = Category, y = MeanIncome, fill = Category)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Mean Income by Category", x = "Category", y = "Mean Income")

ggplot(category_summary, aes(x = Category, y = MeanSpendingScore, fill = Category)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Mean Spending Score by Category", x = "Category", y = "Mean Spending Score")

# Save the cleaned dataset to a CSV file
write.csv(data, "cleaned_data.csv", row.names = FALSE)
cat("Cleaned data saved to 'cleaned_data.csv'.\n")

# End of script
