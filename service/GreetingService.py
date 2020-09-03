
class GreetingService():

    def __init__(self):
        return None

    def sayHello(self):
        return 'Hello From Service'

## 1)model_data_generation

    def lag_decision(self,df_feature_names3):
        if Lag == "L1-L2":
            sets = ["L0", "L1", "L2"]
            return self.df_feature_names3[np.isin(df_feature_names3["lag_type"], sets)]
        elif Lag == "L1-L4":
            sets = ["L0", "L1", "L2", "L3", "L4"]
            return self.df_feature_names3[np.isin(df_feature_names3["lag_type"], sets)]

## 2)lasso_CV

    def scenario_decision(self,scenario):
        if scenario == 1: ## 1 == yes
            return self.pd.Series(["target_perc4Q", "Zincirlenmiş_GDP | perc_4Q | L0"  # ,
                                  # "Zincirlenmiş_GDP | perc_4Q | L1" hatta perc_1Q da eklenebilir
                                  ])
        else:
            return self.pd.Series(["target_perc4Q"])

    def data_preprocessing(self,df_name):
        new_index = missing.loc[missing["model_name"] == model_name, "index"].values[0]
        df_ = df_name.loc[((df_name.index > new_index) & (df_name.index < "2020_Q3")),]  # changed
        df_ = df_.replace([np.inf, -np.inf], np.nan)
        df_ = df_.dropna(axis=0)

        return self.df_

    def train_test_splitting(self,X,y):
        # train test splitting
        X1 = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"], invert=True)]
        y1 = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"], invert=True)].values.reshape(-1, 1)

        X1_test = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"])]
        y1_test = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"])].values.reshape(-1, 1)

        return self.X1,self.y1,self.X1_test,self.y1_test

    def getting_predictions(self,X1,y1):

            pred_train = lasso_regressor.best_estimator_.predict(X1)
            y_lasso_train = pd.DataFrame(data=y1, columns=["Actual"])
            y_lasso_train["year_quarter"] = X1.index
            pred_train2 = pd.DataFrame(data=pred_train, columns=["Pred"])
            pred_train2["year_quarter"] = X1.index
            y_lasso = pd.merge(y_lasso_train, pred_train2, how="left", right_on="year_quarter",
                                     left_on="year_quarter")

            return self.y_lasso


# 4) model_analyses


    def chosen_params(self,all_pred_):
        df = all_pred_.loc[
            ((all_pred_["model_name"] == chosen.loc[i, "names"])
             & (all_pred_["sets"] == chosen.loc[i, "sets"])
             & (all_pred_["chosen_no_of_features_Q"] == chosen.loc[i, "chosen_no_of_features_Q"])
             & (all_pred_["opt_no_of_features_Q"] == chosen.loc[i, "opt_no_of_features_Q"]))
            , ["Actual", "Pred", "year_quarter"]]
        return self.df

    def chosen_lasso(self,all_lasso_coeff_):
       df = all_lasso_coeff_.loc[((all_lasso_coeff_["model_name"] == chosen.loc[i, "names"])
                              & (all_lasso_coeff_["lasso_coeff"] != 0) &
                              (all_lasso_coeff_["sets"] == chosen.loc[i, "sets"])
                              & (all_lasso_coeff_["chosen_no_of_features_Q"] ==
                                 chosen.loc[i, "chosen_no_of_features_Q"])),
                             ["feature", "lasso_coeff"]].sort_values(by="lasso_coeff", ascending=False)
       return self.df

    def plot_figure(self,df):
        plt.figure(figsize=(15, 6))
        plt.plot(figsize=(15, 6), linewidth=2)
        plt.plot(df["year_quarter"], df["Actual"], linewidth=2, color="b", label="actual")
        plt.plot(df["year_quarter"], df["Pred"], linewidth=2, color='r', label="prediction")
        plt.legend(loc="upper right", fontsize="large")
        plt.xticks(df["year_quarter"], rotation='vertical')
        self.plt.show()


