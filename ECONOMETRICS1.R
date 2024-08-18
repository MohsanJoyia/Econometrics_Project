#Loading Libraies
library(dplyr)
library(tidyverse)
library(ggplot2)
library(caret)
library(lmtest)
library(broom)
library(gridExtra)
library(GGally)

##Loading Dataset
data <- read.csv("C://Users//PMLS//Downloads//archive (13)//ecom_customers.csv")
head(data)

names(data)
selected_data <- data[, -(1:3)]
# Fit the multiple linear regression model
model <- lm(`Yearly.Amount.Spent` ~ `Avg..Session.Length` + `Time.on.Website` + `Time.on.App` + `Length.of.Membership`, data = selected_data)

# Summary of the model
summary(model)

#CHECKING ASSUMPTIONS



# 1.Linearity

library(ggplot2)
library(gridExtra)

plot1 <- ggplot(selected_data, aes(x = Avg..Session.Length, y = Yearly.Amount.Spent)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Avg. Session Length vs Yearly Amount Spent") +
  theme_minimal()

plot2 <- ggplot(selected_data, aes(x = Time.on.Website, y = Yearly.Amount.Spent)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Time on Website vs Yearly Amount Spent") +
  theme_minimal()

plot3 <- ggplot(selected_data, aes(x = Time.on.App, y = Yearly.Amount.Spent)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Time on App vs Yearly Amount Spent") +
  theme_minimal()

plot4 <- ggplot(selected_data, aes(x = Length.of.Membership, y = Yearly.Amount.Spent)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Length of Membership vs Yearly Amount Spent") +
  theme_minimal()

grid.arrange(plot1, plot2, plot3, plot4, nrow = 2, ncol = 2)




# 2.X values are fixed in repeated sampling

residuals <- resid(model)
std_residuals <- rstandard(model)
plot_data <- data.frame(`Avg..Session.Length` = selected_data$`Avg..Session.Length`,
                        `Time.on.Website` = selected_data$`Time.on.Website`,
                        `Time.on.App` = selected_data$`Time.on.App`,
                        `Standardized.Residuals` = std_residuals)
pairs(plot_data, 
      main = "Scatter Plot Matrix of Independent Variables and Standardized Residuals",
      pch = 19,    # Use filled circles for points
      cex = 0.7)   # Set point size
cor_matrix <- round(cor(plot_data), 2)
text(0.5, 0.95, paste("Correlation Coefficients:\n", as.matrix(cor_matrix)), 
     cex = 0.7, font = 2)
for (i in 1:3) {
  for (j in 1:3) {
    if (i != j) {
      abline(lm(plot_data[, j] ~ plot_data[, i]), col = "blue")
    }
  }
}

# 3. Normality of residuals

qqnorm(residuals)
qqline(residuals)
title("Normal Q-Q Plot of Residuals")

hist(residuals, breaks = 20, main = "Histogram of Residuals", xlab = "Residuals")
ks.test(residuals, "pnorm", mean = mean(residuals), sd = sd(residuals))
shapiro.test(residuals)



# 4. Zero mean value of disturbance term ɛ

residuals <- residuals(model)
mean_residuals <- mean(residuals)
print(mean_residuals)

median_residuals<- median(residuals)
print(median_residuals)



# 5. asumption Homoscedasticity or equal variance of ɛi
residuals <- residuals(model)
fitted_values <- fitted(model)
plot(fitted_values, residuals,
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values",
     ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)  # Add a horizontal line at y = 0 for reference



# 6. No autocorrelation between the disturbances: E( ɛi,ɛj ) = 0

dw_test <- dwtest(model)
print(dw_test)
# Obtain summary of the model
summary_model <- summary(model)

# Extract R
R_squared <- summary_model$r.squared

# Extract Adjusted R Square
adjusted_R_squared <- summary_model$adj.r.squared

# Extract Std. Error of the Estimate
std_error_estimate <- summary_model$sigma

# Calculate Durbin-Watson statistic
dw_statistic <- dwtest(model)$statistic

# Print the results
cat("R-squared:", R_squared, "\n")
cat("Adjusted R-squared:", adjusted_R_squared, "\n")
cat("Std. Error of the Estimate:", std_error_estimate, "\n")
cat("Durbin-Watson statistic:", dw_statistic, "\n")



# 7. Zero covariance between ɛi and Xi : E ( ɛi,Xi ) = 0

# Fit your linear regression model
model <- lm(Yearly.Amount.Spent ~ Avg..Session.Length + Time.on.Website + Time.on.App + Length.of.Membership, data = selected_data)

# Extract residuals
residuals <- residuals(model)

# Extract predictor variables
predictors <- selected_data[, c("Avg..Session.Length", "Time.on.Website", "Time.on.App", "Length.of.Membership")]

# Scatter plot: Residuals vs each predictor
par(mfrow = c(2, 2))  # Set up a 2x2 grid for plotting

for (i in 1:4) {
  plot(predictors[, i], residuals, 
       xlab = colnames(predictors)[i], ylab = "Residuals",
       main = paste("Residuals vs", colnames(predictors)[i]))
  abline(lm(residuals ~ predictors[, i]), col = "blue")  # Add a linear regression line
}

# Reset plot layout
par(mfrow = c(1, 1))

# Correlation analysis: Calculate correlation coefficients
correlation_coeffs <- sapply(predictors, function(x) cor(residuals, x))
print(correlation_coeffs)

# 6. There is no perfect multicollinearity

# Load necessary libraries
library(car)

# Calculate VIF for each independent variable
vif_values <- vif(model)

# Print VIF values
print(vif_values)

# Interpret VIF values
cat("\nInterpretation of VIF values:\n")
for (i in seq_along(vif_values)) {
  if (vif_values[i] > 10) {
    cat(sprintf("VIF for '%s' is %.2f - indicates high multicollinearity\n", names(vif_values)[i], vif_values[i]))
  } else if (vif_values[i] > 5) {
    cat(sprintf("VIF for '%s' is %.2f - multicollinearity may be a concern\n", names(vif_values)[i], vif_values[i]))
  } else {
    cat(sprintf("VIF for '%s' is %.2f - multicollinearity is not a major concern\n", names(vif_values)[i], vif_values[i]))
  }
}


#### GOODNESS OF FIT

# Extract R-squared and Adjusted R-squared
rsquared <- summary(model)$r.squared
rsquared_adj <- summary(model)$adj.r.squared

# Standard Error of the Estimate
std_error <- summary(model)$sigma

# ANOVA (F-test) for overall significance
anova_result <- anova(model)

# Print results
cat("R-squared: ", rsquared, "\n")
cat("Adjusted R-squared: ", rsquared_adj, "\n")
cat("Standard Error of the Estimate: ", std_error, "\n")
cat("\nANOVA (F-test):\n")
cat("F-statistic: ", anova_result$`F value`[1], "\n")
cat("p-value: ", anova_result$`Pr(>F)`[1], "\n")

# Residual analysis (you can expand this part further if needed)
par(mfrow = c(2, 2))  # Arrange plots in a 2x2 grid
plot(model)  # Generates residual vs. fitted, Q-Q plot, scale-location, and residuals vs. leverage plots
