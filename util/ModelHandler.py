import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from operator import itemgetter
from sklearn import preprocessing
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import Lasso, Ridge
from openpyxl import load_workbook
from os import listdir
from os.path import isfile, join
from service.GreetingService import GreetingService

import warnings

GreetingService = GreetingService()

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class ModelHandler():
    # %%

    def __init__(self):
        return None

    def model_data_generation(self, target_name, target_feature, df_features_, df_ind_, df_target_, df_param_, index,
                              feature_type_set):

        # girdileri fonksiyon içinde çalıştırmak için kopyasını yaratıyoruz
        df_features = df_features_.copy(deep=True)
        df_features = df_features.loc[((df_features.index < "2020_Q3")),]  # changed
        df_target = df_target_.copy(deep=True)
        df_target = df_target.loc[((df_target.index < "2020_Q3")),]  # changed
        df_param = df_param_.copy(deep=True)
        df_ind = df_ind_.copy(deep=True)
        feature_set = feature_type_set

        df_ind2 = df_ind[(df_ind[target_name] == 1)]
        # atıcaz alttakini, üstteki kalacak
        # df_ind2=df_ind[(df_ind[target_name]==1)&(np.isin(df_ind["data"],["Konut_kredileri_Miktar","Konut_kredileri_Kişi","Taşıt_kredileri_miktarı",
        # "Bütçe_açığı_GDP_oranı","Otomotiv_firma_kapasitesi"],invert=True))]
        df_ind2_list = df_ind2["data"].tolist()

        # değişken adlarını parçalara ayırıyoruz
        # sektör için belirlenen indikatör değişkenlerini seçiyoruz
        df_feature_names = pd.DataFrame(df_features.columns, columns=["feature_name"])
        df_feature_names["ind_name_"] = df_feature_names["feature_name"].apply(lambda x: x.split(" | ")[0])
        df_feature_names["lag_type"] = df_feature_names["feature_name"].apply(lambda x: x.split(" | ")[2])
        df_feature_names["feature_type"] = df_feature_names["feature_name"].apply(lambda x: x.split(" | ")[1])
        df_feature_names = df_feature_names[np.isin(df_feature_names["ind_name_"], df_ind2_list)]
        # df_feature_names.to_excel("name.xlsx",sheet_name="1")

        # dictionaryde yer alan parametreleri fonksiyon içindeki değişkenlere alıyoruz
        Lag = df_param.loc[index, "Lag"]

        # ind name e göre yanlarına min_date'i getireceğiz
        df_feature_names2 = pd.merge(df_feature_names, df_ind2[["data", "Dahil Edilebilecek Model"]], how="left",
                                     left_on="ind_name_", right_on="data")

        df_feature_names3_temp = df_feature_names2[
            np.isin(df_feature_names2["feature_name"], ["Zincirlenmiş_GDP | perc_4Q | L0",
                                                        "Zincirlenmiş_GDP | perc_4Q | L0",
                                                        "Zincirlenmiş_GDP | raw | L0"], invert=True)]

        df_feature_names3 = df_feature_names3_temp[(df_feature_names3_temp["lag_type"] == "L0") |
                                                   (df_feature_names3_temp["Dahil Edilebilecek Model"] == "L1") |
                                                   ((df_feature_names3_temp["Dahil Edilebilecek Model"] == "L2") &
                                                    (np.isin(df_feature_names3_temp["lag_type"], ["L0", "L1"],
                                                             invert=True)))]

        # Model içine alınacak laglerin kararını veriyoruz (alınabilir olmasından bağımsız)
        """"
        if Lag == "L1-L2":
            sets = ["L0", "L1", "L2"]
            df_feature_names4 = df_feature_names3[np.isin(df_feature_names3["lag_type"], sets)]
        elif Lag == "L1-L4":
            sets = ["L0", "L1", "L2", "L3", "L4"]
            df_feature_names4 = df_feature_names3[np.isin(df_feature_names3["lag_type"], sets)]
        """

        df_feature_names4 = GreetingService.lag_decision(df_feature_names3, Lag)

        # değişken tipi kararını başta veriyoruz
        df_feature_names5 = df_feature_names4[(np.isin(df_feature_names4["feature_type"], feature_set))]
        df_feature_names5_list = df_feature_names5["feature_name"].tolist()

        # listeden en son isimleri vererek değişkenleri seçelim
        df_features2 = df_features[df_feature_names5_list]

        # targetı seçiyoruz, isim değiştiriyoruz
        target_data = df_target[[target_feature]]
        target_data.rename(columns={target_feature: "target_perc4Q"}, inplace=True)

        # feature ve targetları birleştirerek veri seti haline getiriyoruz
        df_model = pd.merge(df_features2, target_data, how="left", right_index=True, left_index=True)

        return df_model

    def lasso_CV(self, df_new, model_name, missing, features, number_of_features, s, sets_df_, scenario):

        # senaryo yaratılmak istendiğinde buradan yeni değişken eklenebilir
        """""
        if scenario == "yes":
            new_feat = pd.Series(["target_perc4Q", "Zincirlenmiş_GDP | perc_4Q | L0"  # ,
                                  # "Zincirlenmiş_GDP | perc_4Q | L1" hatta perc_1Q da eklenebilir
                                  ])
        else:
            new_feat = pd.Series(["target_perc4Q"])
        """
        new_feat = GreetingService.scenario_decision(scenario)

        features2 = features.append(new_feat)
        df_temp1 = df_new[model_name]
        df_temp2 = df_temp1[features2].copy(deep=True)

        # data preprocessing
        """""
        new_index = missing.loc[missing["model_name"] == model_name, "index"].values[0]
        df_ = df_temp2.loc[((df_temp2.index > new_index) & (df_temp2.index < "2020_Q3")),]  # changed
        df_ = df_.replace([np.inf, -np.inf], np.nan)
        df_ = df_.dropna(axis=0)
        """
        df_ = GreetingService.data_preprocessing(df_temp2, missing, model_name)

        # target data, input data splittin
        X = df_.drop(["target_perc4Q"], axis=1)
        y = df_["target_perc4Q"]

        # train test splitting
        """"
        X1 = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"], invert=True)]
        y1 = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"], invert=True)].values.reshape(-1, 1)

        X1_test = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"])]
        y1_test = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"])].values.reshape(-1, 1)
        """
        X1, y1, X1_test, y1_test = GreetingService.train_test_splitting(X, y, sets_df_, s)

        # lasso model fit
        lasso = Lasso(fit_intercept=False, max_iter=50000)
        parameters = {"alpha": [1e-8, 1e-4, 1e-3, 1e-2, 1, 5]}
        lasso_regressor = GridSearchCV(lasso, parameters, scoring="neg_mean_squared_error", cv=6)
        lasso_regressor.fit(X1, y1)

        # saving model results
        lasso_coeffs = pd.DataFrame(data=lasso_regressor.best_estimator_.coef_, columns=["lasso_coeff"])
        cols = pd.DataFrame(data=X.columns, columns=["feature"])
        coef_lasso = pd.merge(lasso_coeffs, cols, right_index=True, left_index=True)

        alpha = lasso_regressor.best_estimator_.alpha

        # getting predictions
        """""
        pred_train = lasso_regressor.best_estimator_.predict(X1)
        y_lasso_train = pd.DataFrame(data=y1, columns=["Actual"])
        y_lasso_train["year_quarter"] = X1.index
        pred_train2 = pd.DataFrame(data=pred_train, columns=["Pred"])
        pred_train2["year_quarter"] = X1.index
        y_lasso_train = pd.merge(y_lasso_train, pred_train2, how="left", right_on="year_quarter",
                                 left_on="year_quarter")

        pred_test = lasso_regressor.best_estimator_.predict(X1_test)
        y_lasso_test = pd.DataFrame(data=y1_test, columns=["Actual"])
        y_lasso_test["year_quarter"] = X1_test.index
        pred_test2 = pd.DataFrame(data=pred_test, columns=["Pred"])
        pred_test2["year_quarter"] = X1_test.index
        y_lasso_test = pd.merge(y_lasso_test, pred_test2, how="left", right_on="year_quarter", left_on="year_quarter")
        """
        y_lasso_train = GreetingService.getting_predictions(X1, y1, lasso_regressor)
        y_lasso_test = GreetingService.getting_predictions(X1_test, y1_test, lasso_regressor)

        y_lasso_all = pd.concat([y_lasso_train, y_lasso_test], ignore_index=True)

        # calculating error metrics
        rmse_train = np.sqrt(mean_squared_error(y_lasso_train["Actual"], y_lasso_train["Pred"]))
        rmse_test = np.sqrt(mean_squared_error(y_lasso_test["Actual"], y_lasso_test["Pred"]))

        # saving model parameters
        y_lasso_all["chosen_no_of_features_Q"] = number_of_features
        coef_lasso["chosen_no_of_features_Q"] = number_of_features
        y_lasso_all["sets"] = s
        coef_lasso["sets"] = s

        return y_lasso_all, coef_lasso, rmse_train, rmse_test, alpha

    def ridge_trial(self, df, missing, model_name, number_of_features, s, sets_df_, scenario):

        # Data preprocessing
        """""
        new_index = missing.loc[missing["model_name"] == model_name, "index"].values[0]
        df_ = df[model_name].loc[((df[model_name].index > new_index) & (df[model_name].index < "2020_Q3")),]  # changed
        df_ = df_.replace([np.inf, -np.inf], np.nan)
        df_ = df_.dropna(axis=0)
        """
        df_ = GreetingService.data_preprocessing(df[model_name], missing, model_name)

        # target, input data spitting
        X = df_.drop(["target_perc4Q"], axis=1)
        y = df_["target_perc4Q"]

        # train test set splitting
        """
        X1 = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"], invert=True)]
        y1 = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"], invert=True)].values.reshape(-1, 1)

        X1_test = X[np.isin(X.index.values, sets_df_.iloc[s]["sets"])]
        y1_test = y[np.isin(y.index.values, sets_df_.iloc[s]["sets"])].values.reshape(-1, 1)
        """
        X1, y1, X1_test, y1_test = GreetingService.train_test_splitting(X, y, sets_df_, s)

        # recursive feature elimination model fit
        # model estimator is chosen as Ridge
        estimator = Ridge(fit_intercept=False, max_iter=50000)
        selector = RFECV(estimator, scoring="neg_mean_squared_error", step=1, cv=6)
        selector.fit(X1, y1.ravel())
        opt_number_of_features = selector.n_features_

        selector = RFECV(estimator, scoring="neg_mean_squared_error", step=1, cv=6,
                         min_features_to_select=number_of_features)
        selector.fit(X1, y1.ravel())

        # Sonuçların dataframe e kaydedilmesi
        rank = pd.DataFrame(data=selector.ranking_, columns=["ranking"])
        col = pd.DataFrame(data=X.columns, columns=["col_name"])
        rank2 = pd.merge(rank, col, left_index=True, right_index=True)
        rank2["chosen_no_of_features_Q"] = number_of_features
        rank2["sets"] = sets_df_.loc[sets_df_.index == s, "sets"]
        ranking = rank2.sort_values(by="ranking")

        # kritik olarak seçilen değişken isimlerinin features olarak kaydedilmesi (lasso modeline vermek üzere)
        features = rank2[rank2["ranking"] == 1]["col_name"]

        y_lasso_all, lasso_coef_df, rmse_train, rmse_test, alpha_ = self.lasso_CV(df, model_name, missing, features,
                                                                                  number_of_features, s, sets_df_,
                                                                                  scenario)

        return y_lasso_all, ranking, opt_number_of_features, rmse_train, rmse_test, lasso_coef_df, alpha_

    # sonuç excellerini ve model sayısını alır, parametrelere göre en iyi modelleri görselleştirir, bilgilerini yazdırır
    def model_analyses(self, results_, all_pred, all_lasso_coeff, select_first_x):
        results = results_.copy(deep=True)
        all_lasso_coeff_ = all_lasso_coeff.copy(deep=True)
        all_pred_ = all_pred.copy(deep=True)

        # Model sonuçlarından görmek ve karşılaştırmak istediklerimizi metrikleri filtreleyerek seçiyoruz
        # duruma göre değiştirilebilir
        results2 = results[(results["rmse_overfit_metric"] <= 3)
                           & (results["lasso_coeff_number"] >= 3)
                           & (results["yoy_cat_sucess_test"] > 0.6)
                           & (results["yoy_cat_success_all"] > 0.6)
                           & (results["lasso_coeff_number"] <= 10)].sort_values(by=["yoy_cat_success_all", "rmse_test"],
                                                                                ascending=[False, True])

        # yukarıdaki metriklere uyanlardan ilk n tanesini görselleştirme için alıyoruz.
        chosen = results2[:select_first_x]
        chosenlist = chosen.index.tolist()

        # n modeli seçilen parametreleri ve bilgileriyle görselleştiriyoruz
        for i in chosenlist:
            """
            df = all_pred_.loc[
                ((all_pred_["model_name"] == chosen.loc[i, "names"])
                 & (all_pred_["sets"] == chosen.loc[i, "sets"])
                 & (all_pred_["chosen_no_of_features_Q"] == chosen.loc[i, "chosen_no_of_features_Q"])
                 & (all_pred_["opt_no_of_features_Q"] == chosen.loc[i, "opt_no_of_features_Q"]))
                , ["Actual", "Pred", "year_quarter"]]
            """
            df = GreetingService.chosen_params(all_pred_,chosen,i)

            print("Model name: %s" % chosen.loc[i,])
            print("Lasso Coeffs:")
            """""
            print(all_lasso_coeff_.loc[((all_lasso_coeff_["model_name"] == chosen.loc[i, "names"])
                                        & (all_lasso_coeff_["lasso_coeff"] != 0) &
                                        (all_lasso_coeff_["sets"] == chosen.loc[i, "sets"])
                                        & (all_lasso_coeff_["chosen_no_of_features_Q"] ==
                                           chosen.loc[i, "chosen_no_of_features_Q"])),
                                       ["feature", "lasso_coeff"]].sort_values(by="lasso_coeff", ascending=False))
            """
            print(GreetingService.chosen_lasso(all_lasso_coeff_, chosen, i))

            #df.sort_values(by=["year_quarter"], ascending=True, inplace=True)
            #array = df["Pred"].to_numpy()
            #print(array)
            #print(df)
            """"
            df.sort_values(by=["year_quarter"], ascending=True, inplace=True)
            plt.figure(figsize=(15, 6))
            plt.plot(figsize=(15, 6), linewidth=2)
            plt.plot(df["year_quarter"], df["Actual"], linewidth=2, color="b", label="actual")
            plt.plot(df["year_quarter"], df["Pred"], linewidth=2, color='r', label="prediction")
            plt.legend(loc="upper right", fontsize="large")
            plt.xticks(df["year_quarter"], rotation='vertical')
            plt.show()
            """
            GreetingService.plot_figure(df)

    # Fonksiyon: target+featurelardan oluşan dataframe listesini belirtilen quarterı,
    # belirlenen # kritik featurelarla test etmek için alır
    # her bir dataframei ridge_trial fonksiyonuna sokarak prediction, ranking, opt_no_of_features, rmse_train ve rmse_test
    # sonuçlarını verir
    # her model için

    def comparison(self, df_, params_, missing_2, number_of_features, path, target_name, select_first_x, sets,
                   scenario):
        df = df_.copy()
        params = params_.copy(deep=True)
        missing_2_ = missing_2.copy(deep=True)
        params.set_index("Lag", inplace=True, drop=True)
        sets_df = sets

        # modellerin karşılaştırılacağı metrikler için boş arrayler oluşturduk
        opt_no_of_features_Q = []
        chosen_no_of_features_Q = []
        rmse_train_Q = []
        rmse_test_Q = []
        rmse_overfit_metric = []
        lasso_coeff_number = []
        pred_sign_success = []
        pred_mgt_success = []
        yoy_cat_success_train = []
        yoy_cat_sucess_test = []
        yoy_cat_success_all = []
        alpha = []
        names = []
        train_first_index = []
        sets_Q = []

        # feature ranking, tahmin ve feature ağırlıklarının kaydedileceği dataframeleri oluşturuyoruz
        all_rank = pd.DataFrame(
            columns=["target_name", "model_name", "chosen_no_of_features_Q", "ranking", "col_name", "sets"])
        all_pred = pd.DataFrame(
            columns=["target_name", "model_name", "chosen_no_of_features_Q", "year_quarter", "Actual", "pred", "sets"])
        all_lasso_coeff = pd.DataFrame(
            columns=["target_name", "model_name", "chosen_no_of_features_Q", "lasso_coeff", "sets"])

        # her model, train-test zaman aralığı ve RFCV minimum feature sayısı için önce RFECV - ridge modeline vererek
        # veri setini belirleyeceğiz, veri setini lasso modeline vererek en iyi modeli seçeceğiz
        i = 0
        a_list = list(df.keys())

        for i in a_list:
            for s in range(len(sets_df.index)):
                # print (s)
                for k in number_of_features:
                    names.append(i)

                    prediction, ranking, opt_no_of_features, rmse_train_, rmse_test_, lasso_coef_, alpha_ = self.ridge_trial(
                        df, missing_2_, i, k, s, sets_df, scenario)

                    # model varyasyonlarından gelen data kaydediliyor
                    prediction["target_name"] = target_name
                    ranking["target_name"] = target_name
                    lasso_coef_["target_name"] = target_name

                    prediction["model_name"] = i
                    ranking["model_name"] = i
                    lasso_coef_["model_name"] = i

                    prediction["sets"] = s
                    ranking["sets"] = s
                    lasso_coef_["sets"] = s

                    prediction["chosen_no_of_features_Q"] = k
                    ranking["chosen_no_of_features_Q"] = k
                    lasso_coef_["chosen_no_of_features_Q"] = k

                    prediction["opt_no_of_features_Q"] = opt_no_of_features
                    ranking["opt_no_of_features_Q"] = opt_no_of_features
                    lasso_coef_["opt_no_of_features_Q"] = opt_no_of_features
                    lasso_coef_["lasso_coeff_group"] = lasso_coef_["feature"].apply(lambda x: x.split(" | ")[0])

                    # tüm model verilerini tek df'te topluyoruz
                    all_rank = pd.concat([all_rank, ranking], axis=0, ignore_index=True)
                    all_pred = pd.concat([all_pred, prediction], axis=0, ignore_index=True)
                    all_lasso_coeff = pd.concat([all_lasso_coeff, lasso_coef_], axis=0, ignore_index=True)

                    opt_no_of_features_Q.append(opt_no_of_features)
                    chosen_no_of_features_Q.append(k)
                    rmse_train_Q.append(rmse_train_)
                    rmse_test_Q.append(rmse_test_)
                    rmse_overfit_metric.append(rmse_test_ / rmse_train_)
                    sets_Q.append(s)

                    # 5 yeni model karşılaştırma metriği hesaplıyoruz
                    # pred_mgt_success: tahmin ve gerçekleşenin <-%10,-%10< <%10,>%10 3 kategoriden birinde aynı anda olma oranını hesaplar
                    # pred_sign_success: tahmin ve gerçekleşenin <0,>0 2 kategoriden birinde aynı anda olma oranını hesaplar
                    # yoy_cat_success_test: test setinde tahmin ve gerçekleşenin bir önceki tahminden ve gerçekleşenden farkının <-%1,-%1< <%1,>%1 3 kategoriden birinde aynı anda olma oranını hesaplar
                    # yoy_cat_success_train:train setinde tahmin ve gerçekleşenin bir önceki tahminden ve gerçekleşenden farkının <%1,-%1< <%1,>%1 3 kategoriden birinde aynı anda olma oranını hesaplar
                    # yoy_cat_success_all:test + train setinde tahmin ve gerçekleşenin bir önceki tahminden ve gerçekleşenden farkının <%1,-%1< <%1,>%1 3 kategoriden birinde aynı anda olma oranını hesaplar

                    pred_sign = prediction.copy(deep=True)
                    pred_sign["Actual_lag1"] = pred_sign["Actual"].shift(1)
                    pred_sign["Pred_lag1"] = pred_sign["Pred"].shift(1)
                    pred_sign["Actual_yoydiff"] = pred_sign["Actual"] - pred_sign["Actual_lag1"]
                    pred_sign["Pred_yoydiff"] = pred_sign["Pred"] - pred_sign["Actual_lag1"]
                    pred_sign["Actual_yoycat"] = pred_sign["Actual_yoydiff"].apply(
                        lambda x: -1 if x < -0.01 else (1 if x > 0.01 else 0))
                    pred_sign["Pred_yoycat"] = pred_sign["Pred_yoydiff"].apply(
                        lambda x: -1 if x < -0.01 else (1 if x > 0.01 else 0))

                    yoy_cat_success_train.append((pred_sign[(pred_sign["Actual_yoycat"] == pred_sign["Pred_yoycat"]) &
                                                            (np.isin(pred_sign["year_quarter"].values,
                                                                     sets_df.iloc[s]["sets"], invert=True))][
                                                      "Actual"].count())
                                                 / pred_sign[(
                        np.isin(pred_sign["year_quarter"].values, sets_df.iloc[s]["sets"], invert=True))][
                                                     "Actual_yoycat"].count())
                    yoy_cat_sucess_test.append((pred_sign[(pred_sign["Actual_yoycat"] == pred_sign["Pred_yoycat"]) &
                                                          (np.isin(pred_sign["year_quarter"].values,
                                                                   sets_df.iloc[s]["sets"]))]["Actual"].count())
                                               / pred_sign[(
                        np.isin(pred_sign["year_quarter"].values, sets_df.iloc[s]["sets"]))]["Actual_yoycat"].count())
                    yoy_cat_success_all.append(
                        (pred_sign[pred_sign["Actual_yoycat"] == pred_sign["Pred_yoycat"]]["Actual"].count() - 1)
                        / (pred_sign["Actual_yoycat"].count() - 1))

                    pred_sign["Actual_sign"] = pred_sign["Actual"].apply(lambda x: -1 if x < 0 else 1)
                    pred_sign["pred_sign"] = pred_sign["Pred"].apply(lambda x: -1 if x < 0 else 1)
                    pred_sign["Actual_mgt"] = pred_sign["Actual"].apply(
                        lambda x: -1 if x < -0.1 else (1 if x > 0.1 else 0))
                    pred_sign["pred_mgt"] = pred_sign["Pred"].apply(lambda x: -1 if x < -0.1 else (1 if x > 0.1 else 0))
                    pred_mgt_success.append((pred_sign[(pred_sign["Actual_mgt"] == pred_sign["pred_mgt"]) & (
                            pred_sign["Actual_mgt"] != 0)]["Actual"].count()) / (
                                                pred_sign[pred_sign["Actual_mgt"] != 0]["Actual"].count()))
                    pred_sign_success.append(
                        (pred_sign[pred_sign["Actual_sign"] == pred_sign["pred_sign"]]["Actual"].count()) / (
                            pred_sign["Actual"].count()))

                    pred_sign.to_excel("pred_sign.xlsx", sheet_name="0")
                    lasso_coeff_number_ = lasso_coef_[lasso_coef_["lasso_coeff"] != 0]["lasso_coeff_group"].nunique()
                    lasso_coeff_number.append(lasso_coeff_number_)
                    alpha.append(alpha_)

        # mode sonuçlarının karşılaştırılması için results_ dataseti oluşturuyoruz
        results_ = pd.DataFrame(names, columns=["names"])
        results_["target_name"] = target_name
        results_["opt_no_of_features_Q"] = opt_no_of_features_Q
        results_["chosen_no_of_features_Q"] = chosen_no_of_features_Q
        results_["rmse_train"] = rmse_train_Q
        results_["rmse_test"] = rmse_test_Q
        results_["alpha"] = alpha
        results_["sets"] = sets_Q
        results_["rmse_overfit_metric"] = rmse_overfit_metric
        results_["lasso_coeff_number"] = lasso_coeff_number
        results_["pred_sign_success"] = pred_sign_success
        results_["pred_mgt_success"] = pred_mgt_success

        results_["yoy_cat_sucess_test"] = yoy_cat_sucess_test
        results_["yoy_cat_success_train"] = yoy_cat_success_train
        results_["yoy_cat_success_all"] = yoy_cat_success_all

        # karşılaştırma excellerini path'deki dosyalara yazdırıyoruz
        results_.to_excel(path + target_name + "_results.xlsx", index=False)
        all_rank.to_excel(path + target_name + "_rankall.xlsx", index=False)
        all_pred.to_excel(path + target_name + "_predall.xlsx", index=False)
        all_lasso_coeff.to_excel(path + target_name + "_lassocoeff.xlsx", index=False)

        self.model_analyses(results_, all_pred, all_lasso_coeff, select_first_x)

    # run fonksiyonu input veriler ve parametreleri alarak kendi içinde diğer fonksiyonları çağırır
    # her hedef sektör, tahmin hedefi ve path için prediction döngüsünü çalıştırır ve sonuçları pathe kaydeder

    def run(self, df_features, df_ind, target1, target_df, d, no_of_features, sets, select_first_x, feature_type_set,
            scenario):

        params = pd.DataFrame(data=d)
        # hedef sektör, tahmin hedefi ve pathi loop içinde döndürür
        for i in range(len(target_df.index)):

            target_name = target_df.loc[i, "Sektör"]
            target_feature = target_df.loc[i, "target"]
            path = target_df.loc[i, "path"]

            print("Sektör: %s" % target_name)
            print("Hedef: %s" % target_feature)

            # değişiklik laglere sahip setleri model_data_generation fonsiyonunda tek tek yaratır ve bir dictionary'e kaydeder
            df = {}
            for z in range(len(params.index)):
                df[str(params.loc[z, "Lag"])] = self.model_data_generation(target_name, target_feature, df_features,
                                                                           df_ind,
                                                                           target1, params, z, feature_type_set)

            # dictionary'e kaydedilen datasetleri isimlendirir
            a_list = list(df.keys())
            for k in a_list:
                df[k].name = k

            # oluşturulan datasetlerin nan içermeyen en eski index ve nan içermeyen en yeni index ten kesilmesini sağlar

            model_name = []
            index_name = []
            col_name = []
            for k in a_list:
                for n in list(df[k].index):
                    for c in list(df[k].columns):
                        if np.any(np.isnan(df[k].loc[n, c])) == True:
                            model_name.append(k)
                            index_name.append(n)
                            col_name.append(c)
                            # print(k,n,c)
            missing = pd.DataFrame(model_name, columns=["model_name"])
            missing["index"] = index_name
            missing["col_name"] = col_name
            missing_2 = missing[missing["index"] < "2014_Q1"].groupby(by="model_name", as_index=False)["index"].max()

            self.comparison(df, params, missing_2, no_of_features, path, target_name, select_first_x, sets, scenario)

    def init(self):

        # verinin çekilmesi, verinin envanter formatında olması gerekiyor
        df_features = pd.read_excel("IC_veri_envanteri.xlsx", sheet_name="features")
        df_features.set_index("year_quarter", drop=True, inplace=True)

        # Oluşturulan data explanation tabi çekilir
        # bu excelde ind_name, sektör kolonları (çimento, lastik, enerji, sigorta_x) ve Lag olacak.
        df_ind = pd.read_excel("IC_veri_envanteri.xlsx", sheet_name="envanter")

        # Oluşturulan hedef verisi çekilir ve isimlendirme yapılır
        # autoregressive yapmayacağız yalnızca perc4Q yu hedef olarak alıyoruz. Quarterı değişkenlerden çıkarıyoruz
        target1 = pd.read_excel("IC_veri_envanteri.xlsx", sheet_name="targets")
        target1.set_index("year_quarter", drop=True, inplace=True)

        # gün ve sıcaklık verilerinin ham halinin normalize edilmesi ve ham verilerin bu şekilde güncellenmesi
        collist1 = df_features.columns.tolist()
        col_df = pd.DataFrame(data=collist1, columns=["name"])
        col_df["ind_name"] = col_df["name"].apply(lambda x: x.split(" | ")[0])
        col_df["lag_name"] = col_df["name"].apply(lambda x: x.split(" | ")[2])
        test = ['CHDD_all',
                'CHDD_enerjisa',
                'Kar_örtülü_gün_DoğuAnadolu',
                'Kar_örtülü_gün_Ege',
                'Kar_örtülü_gün_GüneydoğuAnadolu',
                'Kar_örtülü_gün_Karadeniz',
                'Kar_örtülü_gün_Marmara',
                'Kar_örtülü_gün_İçAnadolu',
                'Çeyreklik_ramazan_gün_sayısı',
                'Çeyreklik_tatil_sayısı',
                'Çeyrekteki_gün_sayısı',
                'Çeyrekteki_işgünü_sayısı']
        scaling_list = col_df[(np.isin(col_df["ind_name"], test))]["name"].tolist()

        # gün, sıcaklık verilerinin L0 lagi dışında tüm laglerini features dataframeinden atıyoruz.
        # Kalan raw L0 verilerini normalize ediyoruz ve o şekilde features verisinde güncelliyoruz.
        # elimination=col_df[np.isin(col_df["name"],elimination_list1,invert=True)]
        # elimination_list2=elimination["name"].tolist()
        # df_features=df_features[elimination_list2]
        # col_df3=elimination[np.isin(elimination["ind_name"],test)]
        # col_list=col_df3["name"].tolist()

        max_abs_scaler = preprocessing.MaxAbsScaler()
        df_features[scaling_list] = max_abs_scaler.fit_transform(df_features[scaling_list])

        df_features["CHDD_all | diff_4Q | L0"] = df_features["CHDD_all | raw | L0"] - df_features[
            "CHDD_all | raw | L0"].shift(4)
        df_features["CHDD_enerjisa | diff_4Q | L0"] = df_features["CHDD_enerjisa | raw | L0"] - df_features[
            "CHDD_enerjisa | raw | L0"].shift(4)

        df_features["Kar_örtülü_gün_DoğuAnadolu | diff_4Q | L0"] = df_features[
                                                                       "Kar_örtülü_gün_DoğuAnadolu | raw | L0"] - \
                                                                   df_features[
                                                                       "Kar_örtülü_gün_DoğuAnadolu | raw | L0"].shift(4)
        df_features["Kar_örtülü_gün_Ege | diff_4Q | L0"] = df_features["Kar_örtülü_gün_Ege | raw | L0"] - df_features[
            "Kar_örtülü_gün_Ege | raw | L0"].shift(4)
        df_features["Kar_örtülü_gün_GüneydoğuAnadolu | diff_4Q | L0"] = df_features[
                                                                            "Kar_örtülü_gün_GüneydoğuAnadolu | raw | L0"] - \
                                                                        df_features[
                                                                            "Kar_örtülü_gün_GüneydoğuAnadolu | raw | L0"].shift(
                                                                            4)
        df_features["Kar_örtülü_gün_Karadeniz | diff_4Q | L0"] = df_features["Kar_örtülü_gün_Karadeniz | raw | L0"] - \
                                                                 df_features[
                                                                     "Kar_örtülü_gün_Karadeniz | raw | L0"].shift(4)
        df_features["Kar_örtülü_gün_Marmara | diff_4Q | L0"] = df_features["Kar_örtülü_gün_Marmara | raw | L0"] - \
                                                               df_features["Kar_örtülü_gün_Marmara | raw | L0"].shift(4)
        df_features["Kar_örtülü_gün_İçAnadolu | diff_4Q | L0"] = df_features["Kar_örtülü_gün_İçAnadolu | raw | L0"] - \
                                                                 df_features[
                                                                     "Kar_örtülü_gün_İçAnadolu | raw | L0"].shift(4)

        df_features["Çeyreklik_ramazan_gün_sayısı | diff_4Q | L0"] = df_features[
                                                                         "Çeyreklik_ramazan_gün_sayısı | raw | L0"] - \
                                                                     df_features[
                                                                         "Çeyreklik_ramazan_gün_sayısı | raw | L0"].shift(
                                                                         4)
        df_features["Çeyreklik_tatil_sayısı | diff_4Q | L0"] = df_features["Çeyreklik_tatil_sayısı | raw | L0"] - \
                                                               df_features["Çeyreklik_tatil_sayısı | raw | L0"].shift(4)
        df_features["Çeyrekteki_gün_sayısı | diff_4Q | L0"] = df_features["Çeyrekteki_gün_sayısı | raw | L0"] - \
                                                              df_features["Çeyrekteki_gün_sayısı | raw | L0"].shift(4)
        df_features["Çeyrekteki_işgünü_sayısı | diff_4Q | L0"] = df_features["Çeyrekteki_işgünü_sayısı | raw | L0"] - \
                                                                 df_features[
                                                                     "Çeyrekteki_işgünü_sayısı | raw | L0"].shift(4)

        # %%

        # sign & mgmt kuralı konmadı. max 3
        target_dic = {'Sektör': [  # 'Çimento_tüketimi_ton' , #çimento ayrı yapılacak 2020_q2 si yok
            # "Lastik_satışları_toplamı",
            'Lastik_satışları_Replacement_tüketici_PSR',
            'Lastik_satışları_Replacement_LT_TBR_LSR',
            'Lastik_satışları_OE_tüketici_PSR',
            'Lastik_satışları_OE_LT_TBR_LSR',
            # 'Elektrik_talebi', #en son 2020_Q1 var
            'Hayat_prim',
            # "Kredi_bağlantılı_hayat_prim",
            # "Serbest_hayat_prim",
            'BES_katılımcı',
            # "Hayat_dışı_toplam_prim",
            'Hayat_dışı_motor_prim',
            'Hayat_dışı_nonmotor_prim',
            'Hayat_dışı_sağlık_prim'
        ],
            'target': [  # 'Çimento_tüketimi_ton | perc_4Q', #çimento ayrı yapılacak 2020_q2 si yok
                # 'Lastik_satışları_toplamı | perc_4Q',
                'Lastik_satışları_Replacement_tüketici_PSR | perc_4Q',
                'Lastik_satışları_Replacement_LT_TBR_LSR | perc_4Q',
                'Lastik_satışları_OE_tüketici_PSR | perc_4Q',
                'Lastik_satışları_OE_LT_TBR_LSR | perc_4Q',
                # 'Elektrik_talebi | perc_4Q', #en son 2020_Q1 var
                'Hayat_prim | perc_4Q',
                # 'Kredi_bağlantılı_hayat_prim | perc_4Q',
                # 'Serbest_hayat_prim | perc_4Q',
                'BES_katılımcı | perc_4Q',
                # 'Hayat_dışı_toplam_prim | perc_4Q',
                'Hayat_dışı_motor_prim | perc_4Q',
                'Hayat_dışı_non_motor_prim | perc_4Q',
                'Hayat_dışı_sağlık_prim | perc_4Q'
            ]
            ,
            'path': [
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Modeller_15072020_woscenario - v2//',
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Modeller_15072020_woscenario - v2//',
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Modeller_15072020_woscenario - v2//',
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Modeller_15072020_woscenario - v2//',
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Final IC_12082020//Model Selection & 1st Predictions//Modeller_14082020//',
                # 'C://Users//pinar.yalcin//Desktop//Projects//IndustryCycles//Faz2//Faz2_13032020//Final IC_12082020//Model Selection & 1st Predictions//Modeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//',
                'C://Users//ulas.eraslan//Desktop//Projects//Model Selection & 1st PredictionsModeller_14082020//'
            ]}

        target_df = pd.DataFrame(data=target_dic)

        d = {'Lag': ["L1-L2", "L1-L4"]}

        no_of_features = (5, 10)

        set = ["2011_Q1", "2011_Q2", "2011_Q3", "2011_Q4",
               "2012_Q1", "2012_Q2", "2012_Q3", "2012_Q4",
               "2013_Q1", "2013_Q2", "2013_Q3", "2013_Q4",
               "2014_Q1", "2014_Q2", "2014_Q3", "2014_Q4",
               "2015_Q1", "2015_Q2", "2015_Q3", "2015_Q4",
               "2016_Q1", "2016_Q2", "2016_Q3", "2016_Q4",
               "2017_Q1", "2017_Q2", "2017_Q3", "2017_Q4",
               "2018_Q1", "2018_Q2", "2018_Q3", "2018_Q4",
               "2019_Q1", "2019_Q2", "2019_Q3", "2019_Q4", "2020_Q1", "2020_Q2"]  # 3 data point daha ekledik

        set_name = []
        set_number = []
        set_ = []
        for i in range(0, 50, 1):
            set_number.append(i)
            set_name.append("set" + str(i))
            set_temp = random.sample(set, 6)
            # set_temp.extend(["2019_Q4"])
            set_.append(set_temp)
        sets = {"set_name": set_name, "no": set_number, "sets": set_}

        feature_type_set = ["perc_1Q", "perc_4Q", "diff_1Q", "diff_4Q", "raw"]
        # change_rate_1Q, change_rate_4Q yu da denemede ekleyeceksin
        # raw sıcaklık ve gün verilerinden geliyor

        sets_df = pd.DataFrame(data=sets)
        sets_df.to_excel("setler_14082020_1004.xlsx", sheet_name="1")

        scenario = 0

        select_first_x = 10

        self.run(df_features, df_ind, target1, target_df, d, no_of_features, sets_df, select_first_x, feature_type_set,
                 scenario)
