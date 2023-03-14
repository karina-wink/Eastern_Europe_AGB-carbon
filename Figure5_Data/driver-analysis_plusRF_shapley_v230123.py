# code to identify drivers of AGB carbon change in EE

from osgeo import gdalnumeric
#import os, fnmatch
import numpy as np



# load EsternEuropean extent map
EE_map_ref = gdalnumeric.LoadFile("EUR_extent_quarterdeg_nn.tif")
EUR_east_codes = [27, 57, 63, 97, 119, 126, 146, 167, 173, 183, 185, 199, 230]
EE_map = np.zeros(EE_map_ref.shape, dtype=np.uint16)

for e in EUR_east_codes: 
    EE_map[EE_map_ref == e] = 1


EUR_east_names = ["Bulgaria", "Belarus", "Estonia", "Hungary", "Latvia", "Lithuania", "Moldova", 
                  "Czech Republic", "Poland", "Romania", "Russia", "Slovakia", "Ukraine"]


# load mean AGB change file
mean_agb_change = gdalnumeric.LoadFile('3layer_AGB-C_mean-change-agreemskd_2010-2019_withoutTRENDY.tif')
mean_agbc_class = gdalnumeric.LoadFile('Carbon_sinks-sources_classified_withoutTRENDY.tif')



# load driver component files


############
# LUM (BLUE land use)
lum_BLUE = gdalnumeric.LoadFile('Driver_components/BLUE_eluc_mean-anndiff_2010-2019.nc')
lum_BLUE[EE_map == 0] = np.nan
# standardise 
lum_BLUE_sd = np.full(lum_BLUE.shape, np.nan, dtype=(np.float32))
subset = lum_BLUE[abs(mean_agbc_class) == 1]
lum_BLUE_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)

# # range -1 +1
# lum_BLUE[lum_BLUE>0] = lum_BLUE[lum_BLUE>0]/np.nanmax(lum_BLUE)
# lum_BLUE[lum_BLUE<0] = -lum_BLUE[lum_BLUE<0]/np.nanmin(lum_BLUE)



# LUM (BLUE agr emissions)
lum_BLUE_agr = gdalnumeric.LoadFile('Driver_components/BLUE_eagr_mean-anndiff_2010-2019.nc')
lum_BLUE_agr[EE_map == 0] = np.nan
# standardise 
lum_BLUE_agr_sd = np.full(lum_BLUE_agr.shape, np.nan, dtype=(np.float32))
subset = lum_BLUE_agr[abs(mean_agbc_class) == 1]
lum_BLUE_agr_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)

# range -1 +1
# lum_BLUE_agr[lum_BLUE_agr>0] = lum_BLUE_agr[lum_BLUE_agr>0]/np.nanmax(lum_BLUE_agr)
# lum_BLUE_agr[lum_BLUE_agr<0] = -lum_BLUE_agr[lum_BLUE_agr<0]/np.nanmin(lum_BLUE_agr)

# LUM (BLUE abandonment)
lum_BLUE_abd = gdalnumeric.LoadFile('Driver_components/BLUE_eaban_mean-anndiff_2010-2019.nc')
lum_BLUE_abd[EE_map == 0] = np.nan
# standardise
lum_BLUE_abd_sd = np.full(lum_BLUE_abd.shape, np.nan, dtype=(np.float32))
subset = lum_BLUE_abd[abs(mean_agbc_class) == 1]
lum_BLUE_abd_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# lum_BLUE_abd[lum_BLUE_abd>0] = lum_BLUE_abd[lum_BLUE_abd>0]/np.nanmax(lum_BLUE_abd)
# lum_BLUE_abd[lum_BLUE_abd<0] = -lum_BLUE_abd[lum_BLUE_abd<0]/np.nanmin(lum_BLUE_abd)

# LUM (BLUE harvest)
lum_BLUE_harv = gdalnumeric.LoadFile('Driver_components/BLUE_eharv_mean-anndiff_2010-2019.nc')
lum_BLUE_harv[EE_map == 0] = np.nan
# standardise 
lum_BLUE_harv_sd = np.full(lum_BLUE_harv.shape, np.nan, dtype=(np.float32))
subset = lum_BLUE_harv[abs(mean_agbc_class) == 1]
lum_BLUE_harv_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)

# # range -1 +1
# lum_BLUE_harv[lum_BLUE_harv>0] = lum_BLUE_harv[lum_BLUE_harv>0]/np.nanmax(lum_BLUE_harv)
# lum_BLUE_harv[lum_BLUE_harv<0] = -lum_BLUE_harv[lum_BLUE_harv<0]/np.nanmin(lum_BLUE_harv)

# abandonment (belongs to LUM-land use - sink) 
lum_abandon = gdalnumeric.LoadFile('Driver_components/Estel_Lesiv_Hilda_abandonment_fraction_mean.tif')
lum_abandon[EE_map == 0] = np.nan
# standardise 
lum_abandon_sd = np.full(lum_abandon.shape, np.nan, dtype=(np.float32))
subset = lum_abandon[abs(mean_agbc_class) == 1]
lum_abandon_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# lum_abandon[lum_abandon>0] = lum_abandon[lum_abandon>0]/np.nanmax(lum_abandon)
# lum_abandon[lum_abandon<0] = np.nan


############
# forest harvest (belongs to LUM-land management - source)
lum_harvest = gdalnumeric.LoadFile('Driver_components/Diff_harvestedforest_2010-2019_quarterdeg.tif')
lum_harvest[EE_map == 0] = np.nan


# load additional hildaplus forest map
hildaplus2010 = gdalnumeric.LoadFile('Driver_components/HILDAp_states_2010_mode.tif')
lum_harvest[hildaplus2010 != 44] = 0

# standardise 
lum_harvest_sd = np.full(lum_harvest.shape, np.nan, dtype=(np.float32))
subset = lum_harvest[abs(mean_agbc_class) == 1]
lum_harvest_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)

# # range -1 +1
# lum_harvest[lum_harvest>0] = lum_harvest[lum_harvest>0]/np.nanmax(lum_harvest)
# lum_harvest[lum_harvest<0] = -lum_harvest[lum_harvest<0]/np.nanmin(lum_harvest)



############
# fire (belongs to ENV - source)
env_fire = gdalnumeric.LoadFile('Driver_components/ESACCI_BurnedArea_annualdiff_mean_2010-2019_km2.nc')
env_fire[EE_map == 0] = np.nan
# standardise 
env_fire_sd = np.full(env_fire.shape, np.nan, dtype=(np.float32))
subset = env_fire[abs(mean_agbc_class) == 1]
env_fire_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_fire[env_fire>0] = env_fire[env_fire>0]/np.nanmax(env_fire)
# env_fire[env_fire<0] = -env_fire[env_fire<0]/np.nanmin(env_fire)


############
# temperature (belongs to ENV )
env_temp = gdalnumeric.LoadFile('Driver_components/BerkeleyEarth_Temperature_annualDiff_2010-2019_quarterdeg.tif')
env_temp[env_temp < -1000] = np.nan
env_temp[EE_map == 0] = np.nan
# standardise 
env_temp_sd = np.full(env_temp.shape, np.nan, dtype=(np.float32))
subset = env_temp[abs(mean_agbc_class) == 1]
env_temp_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_temp[env_temp>0] = env_temp[env_temp>0]/np.nanmax(env_temp)
# env_temp[env_temp<0] = -env_temp[env_temp<0]/np.nanmin(env_temp)


# precipitation (belongs to ENV )
env_prec = gdalnumeric.LoadFile('Driver_components/TerraClimatePrecipitation_AnnualSum_Trend_2010-2019_quarterdeg.tif')
env_prec[EE_map == 0] = np.nan
# standardise 
env_prec_sd = np.full(env_prec.shape, np.nan, dtype=(np.float32))
subset = env_prec[abs(mean_agbc_class) == 1]
env_prec_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_prec[env_prec>0] = env_prec[env_prec>0]/np.nanmax(env_prec)
# env_prec[env_prec<0] = -env_prec[env_prec<0]/np.nanmin(env_prec)

# soil moisture (belongs to ENV )
env_sm = gdalnumeric.LoadFile('Driver_components/Copernicus_soilmoisture/soilmoisture_trend_annualmean2010-2019.tif')
env_sm[env_sm < -10000] = np.nan
env_sm[EE_map == 0] = np.nan
# standardise 
env_sm_sd = np.full(env_sm.shape, np.nan, dtype=(np.float32))
subset = env_sm[abs(mean_agbc_class) == 1]
env_sm_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_sm[env_sm>0] = env_sm[env_sm>0]/np.nanmax(env_sm)
# env_sm[env_sm<0] = -env_sm[env_sm<0]/np.nanmin(env_sm)

# CO2  (belongs to ENV )
env_co2 = gdalnumeric.LoadFile('Driver_components/CAMS_GHG_CO2conc_2010-2019_annmean_quarterdeg.tif')
env_co2[EE_map == 0] = np.nan
# standardise 
env_co2_sd = np.full(env_co2.shape, np.nan, dtype=(np.float32))
subset = env_co2[abs(mean_agbc_class) == 1]
env_co2_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_co2[env_co2>0] = env_co2[env_co2>0]/np.nanmax(env_co2)
# env_co2[env_co2<0] = -env_co2[env_co2<0]/np.nanmin(env_co2)

# N deposition  (belongs to ENV )
env_N = gdalnumeric.LoadFile('Driver_components/N_tot_diff_mean2004-06-mean2014-16_quarterdeg_annual_kgN-per-ha.tif')
env_N[EE_map == 0] = np.nan
# standardise 
env_N_sd = np.full(env_N.shape, np.nan, dtype=(np.float32))
subset = env_N[abs(mean_agbc_class) == 1]
env_N_sd[abs(mean_agbc_class) == 1] = (subset-np.nanmean(subset))/np.nanstd(subset)
# # range -1 +1
# env_N[env_N>0] = env_N[env_N>0]/np.nanmax(env_N)
# env_N[env_N<0] = -env_N[env_N<0]/np.nanmin(env_N)




#########################
# DRIVER ANALYSIS



# feature importance/ random forest regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import math
import shap

#plt.ion()

for subset in ["source", "sink", "all"]: 
    dataset = pd.DataFrame({"AGBcarbon": mean_agb_change[EE_map ==1], 
                            "BLUE-aban": lum_BLUE_abd[EE_map == 1],
                            "BLUE-agr": lum_BLUE_agr[EE_map == 1],
                            "BLUE-harv": lum_BLUE_harv[EE_map == 1],
                            "Abandonment": lum_abandon[EE_map == 1], 
                            "Harvest": lum_harvest[EE_map == 1], 
                            "Fire": env_fire[EE_map == 1], 
                            "Temperature": env_temp[EE_map == 1], 
                            "Precipitation": env_prec[EE_map == 1], 
                            "Soil Moisture": env_sm[EE_map == 1], 
                            "CO2": env_co2[EE_map == 1], 
                            "Nitrogen": env_N[EE_map == 1]})
    
    dataset.head()
    dataset = dataset.dropna()

    if subset == "source": 
        dataset = dataset[dataset["AGBcarbon"] < 0]
        print("run driver analysis for ", subset)
    if subset == "sink": 
        dataset = dataset[dataset["AGBcarbon"] > 0]
        print("run driver analysis for ", subset)
    if subset == "all": 
        print("run driver analysis for ", subset)
    
    # preprocess dataset
    X = dataset.iloc[:,1:]
    y = dataset.iloc[:,0]
    
    # fit random forest regressor
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    
    # Scale the data to be between -1 and 1
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns)
    
    # Establish model
    regressor = RandomForestRegressor(n_jobs=-1)
    #regressor = RandomForestRegressor(n_estimators = 100)
    
    
    # Try different numbers of n_estimators - this will take a minute or so
    estimators = np.arange(10, 200, 10)
    scores = []
    for n in estimators:
        regressor.set_params(n_estimators=n)
        regressor.fit(X_train, y_train)
        scores.append(regressor.score(X_test, y_test))
    
    plt.rcdefaults()
    plt.rc('font', size=24) 
    fig, ax = plt.subplots(figsize=(10,8))
    plt.title("Effect of n_estimators")
    plt.xlabel("n_estimator")
    plt.ylabel("score")
    plt.plot(estimators, scores)
    plt.show()
    
    
    n_est = estimators[np.argmax(scores)]
    
    # apply number of estimators with highest score 
    regressor = RandomForestRegressor(n_estimators = n_est)
    
    # fit the model
    regressor.fit(X_train,y_train)
    
    regressor.score(X_train, y_train)
    R_squared = regressor.score(X_test, y_test)
    print('R squared is', round(R_squared, 2)) 
    
    
    predictions = regressor.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    rmse = math.sqrt(mse)
    print('RMSE is',round(100*max(0,rmse)), "%") 
    

    
    # scores = defaultdict(list)
    
    # #crossvalidate the score on a number of different estimators
    # estimators = np.arange(n_est-50, n_est+50, 10)
    # for n in estimators:
    #     print("nr of estimators: ", n)
    #     regressor = RandomForestRegressor(n_estimators = n)

    #     # crossvalidate the scores on a number of different random splits of the data
    #     for _ in range(10):
    #         print(_)
    #         train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.25)
    #         regressor.fit(train_X, train_y)
    #         acc = metrics.r2_score(valid_y, regressor.predict(valid_X))
            
    #         for column in X.columns:
    #             X_t = valid_X.copy() 
    #             X_t[column] = np.random.permutation(X_t[column].values)
    #             shuff_acc = metrics.r2_score(valid_y, regressor.predict(X_t))
    #             scores[column].append(np.abs(acc-shuff_acc)/acc)
            
    # print('Features sorted by their score:')
    
    # print(sorted([(round(np.mean(score), 4), feat) for feat, score in 
    #               scores.items()], reverse=True))
            
    # feat_imp = pd.DataFrame(sorted([(round(np.mean(score), 4), feat) for feat, score in 
    #               scores.items()], reverse=True))    
    
    
    # colors = np.array(["#fb8072", "#ea5cfd", "#fcb962", "#fb8072", "#fcb962", "#ffff80", "#9980bd", "#80b1d3", "#8dd3c7", "#b3de69", "#ccebc5"])

    # sorted_idx = list()
    
    # for i in feat_imp[1]: 
    #     sorted_idx.append(list(dataset.columns.values[1:]).index(i))
    
    
    # plt.rcdefaults()
    # plt.rc('font', size=30) 
    # fig, ax = plt.subplots(figsize=(10,8))
    
    # ax.barh(dataset.columns[1:][sorted_idx][::-1], feat_imp[0][::-1], color = colors[sorted_idx][::-1], align='center')
    # ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Importance (performance decrease)')
    # ax.invert_yaxis()  # labels read top-to-bottom
    # plt.title("$R^2={:.2f}$".format(r2_score(y_test, predictions)) + " (n="+ str(dataset.shape[0]) + ")")
    # #plt.show()
    # plt.savefig("CCarbon_driver_feature_importance_v230123/EasternEurope_carbon-"+subset+"_feature_importance_RF.png", 
    #             bbox_inches = "tight")
    
    
    
    regressor = RandomForestRegressor(n_estimators = n_est)
    regressor.fit(X_train,y_train)
    regressor.score(X_train, y_train)
    R_squared = regressor.score(X_test, y_test)
    #shapley values
    
    
    # load JS visualization code to notebook
    shap.initjs()
    
    # Explain the model’s predictions using shap. Collect the explainer and the shap_values
    
    explainer = shap.TreeExplainer(regressor)
    #shap_values = explainer.shap_values(X_test)
    shap_values0 = explainer(X_test)
    # shap_values0 = explainer(X_train)
    # shap_values0 = explainer(X)

    
    # # Force Plot for one observation only
    # i = 100
    # p = shap.force_plot(explainer.expected_value, explainer.shap_values(X_test[[i]]), features=X.iloc[i], feature_names=X.columns, matplotlib = True, show = False)
    # plt.savefig('Carbon_driver_feature_importance_v230123/EasternEurope_carbon-'+subset+'_shapley_forceplot.png')
    # plt.close() 
    # p
    # # Summary Plot
    
    # shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, show = False, 
    #                   title = "$R^2={:.2f}$".format(r2_score(y_test, predictions)) + " (n="+ str(dataset.shape[0]) + ")")
    # plt.savefig("shapley_summary_all_pixels.png")
    
    
    # plt.rcdefaults()
    # plt.rcParams.update({'font.size': 22})
    # fig = plt.figure()
    
    #shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, show = False) 
    
    shap.plots.beeswarm(shap_values0)
    shap.plots.bar(shap_values0.abs.mean(0))
    
    
    # Summary Bar Plot
    shap.summary_plot(shap_values0, features=X, feature_names=X.columns, plot_type='bar')
    
    
    fig = plt.figure()
    plt.gcf().set_size_inches(20,6)
    shap.plots.bar(shap_values0, show = False)
    plt.show()
    
    shap.plots.heatmap(shap_values0[1:1000])
