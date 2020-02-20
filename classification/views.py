from django.shortcuts import render
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier
import os
import datetime
#Importing data


# Create your views here.
def index(request):
	direc = os.getcwd()+"\classification\Electricity_Data.csv"
	print(direc)
	df = pd.read_csv(direc)
	feature_column = ['Population','Transmitted energy GWH','Net Electricity Consumption Kwh','Lost Electricty Kwh']
	label_column = ['Labels']
	feature_vector = df[feature_column]
	label_vector = df[label_column]
	label_vector['Labels'] = label_vector['Labels'].map({'Poor': 0, 'Low': 1, 'Average':2, 'High':3})
	train  = feature_vector[0:300]
	test = feature_vector[300:420]
	train_label = label_vector[0:300]
	test_label = label_vector[300:420]
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(train,train_label)
	predict = neigh.predict(test)
	accuracy = neigh.score(test,test_label)
	print(predict)

	selected_feat = ['Net Electricity Consumption Kwh','Lost Electricty Kwh','Labels']
	show_data = df[selected_feat]
	data = {'Low':list(),'Poor':list(),'Average':list(),'High':list()}
	for j in range(show_data.shape[0]):
		value = show_data.iloc[j]
		if value['Labels'] not in data:
			data[value['Labels']]=list()
			temp = {'x': value['Net Electricity Consumption Kwh'], 'y':value['Lost Electricty Kwh'] }
			data[value['Labels']].append(temp)
		else:
			temp = {'x': value['Net Electricity Consumption Kwh'], 'y':value['Lost Electricty Kwh'] }
			data[value['Labels']].append(temp)
		print(value['Net Electricity Consumption Kwh'])
	print(data)
	stats= list()
	stats.append({'y':len(data['Poor']), 'label':'Poor'})
	stats.append({'y':len(data['Low']), 'label':'Low'})
	stats.append({'y':len(data['Average']), 'label':'Average'})
	stats.append({'y':len(data['High']), 'label':'High'})
	trainstats=300
	teststats = 120
	total = show_data.shape[0]
	classifier = "K Nearest Neighbour"

	efficiency_vector = df[['Date (Montly)','Transmitted energy GWH','Net Electricity Consumption Kwh']]
	effic = list()
	for k in range(efficiency_vector.shape[0]):
		value = efficiency_vector.iloc[k]
		dt = value['Date (Montly)']
		eff = (int(value['Transmitted energy GWH'])*1000000)/int(value['Net Electricity Consumption Kwh'])*100
		#timestamp=datetime.datetime(int(dt.split('-')[0]), int(dt.split('-')[1]), int(dt.split('-')[2].split(' ')[0]))
		timestamp = dt
		temp = {'x':timestamp, 'y':eff}
		effic.append(temp)
	usage = list()
	for l in range(0,380):
		value = efficiency_vector.iloc[l]
		dt = value['Date (Montly)']
		use = float(value['Net Electricity Consumption Kwh'])
		temp = {'x':dt, 'y':use}
		usage.append(temp)
	temp_usage = list(usage)
	index = 380
	usage_forecast = list()
	for k  in range(0,40):
		sum=0
		for l in range(index-21,index):
			sum+=temp_usage[l]['y']
		sum=sum/21
		t = {'x':effic[index-1]['x'],'y':sum}
		usage_forecast.append(t)
		temp_usage.append(t)	
		index+=1
	
	saving = list()
	for l in range(0,380):
		value = efficiency_vector.iloc[l]
		dt = value['Date (Montly)']
		save = float(int(value['Transmitted energy GWH'])*1000000)-float(value['Net Electricity Consumption Kwh'])
		temp = {'x':dt, 'y':save}
		saving.append(temp)
	temp_saving = list(saving)
	index = 380
	saving_forecast = list()
	for k  in range(0,40):
		sum=0
		for l in range(index-21,index):
			sum+=temp_saving[l]['y']
		sum=sum/21
		t = {'x':effic[index-1]['x'],'y':sum}
		saving_forecast.append(t)
		temp_saving.append(t)	
		index+=1

	

	return render(request, 'dashboard.html',{"data":data, 'stat':stats, 'total':total ,'saving':saving,'saving_forecast':saving_forecast,'usage':usage,'usage_forecast':usage_forecast,'efficiency':effic ,'train':trainstats, 'test':teststats, 'classifier':classifier })

def weather(request):
	direc = os.getcwd()+"\classification\weather_and_gas_data.csv"
	print(direc)
	df = pd.read_csv(direc)
	feature_column = ['pm10','pm10_level','co','co_level','no2','no2_level','pm25','pm25_level','last_measure']
	label_column = ['Labels']
	feature_vector = df[feature_column]
	# label_vector = df[label_column]
	feature_vector['pm10_level'] = feature_vector['pm10_level'].map({'unhealthy-high': 0, 'unhealthy-low': 1,'unhealthy':2, 'moderate':3, 'good':4})
	feature_vector['co_level'] = feature_vector['co_level'].map({'unhealthy-high': 0, 'unhealthy-low': 1,'unhealthy':2, 'moderate':3, 'good':4})
	feature_vector['no2_level'] = feature_vector['no2_level'].map({'unhealthy-high': 0, 'unhealthy-low': 1,'unhealthy':2, 'moderate':3, 'good':4})
	feature_vector['pm25_level'] = feature_vector['pm25_level'].map({'unhealthy-high': 0, 'unhealthy-low': 1,'unhealthy':2, 'moderate':3, 'good':4})
	feature_vector['label'] = (round((feature_vector['pm10_level']+feature_vector['co_level']+feature_vector['no2_level']+feature_vector['pm25_level'])/4,0))
	#feature_vector.set_index(pd.DatetimeIndex(feature_vector['last_measure']))
	#feature_vector['index']=pd.to_datetime(pd.DatetimeIndex(feature_vector['last_measure']))
	feature_vector['index']=feature_vector['last_measure']
	feature_vector = feature_vector.sort_values(['index'])
	print(feature_vector.sort_values(['index']))
	new_feature_column = ['pm10','co','no2','pm25',]
	label_column = ['label']
	new_feature_vector = feature_vector[new_feature_column]
	new_label_vector = feature_vector[label_column]
	new_label_vector['label']= new_label_vector['label'].map({0.00:0, 1.00:1, 2.00:2, 3.00:3, 4.00:4}) 
	train  = new_feature_vector[0:610]
	train_label = new_label_vector[0:610]
	test = new_feature_vector[610:720]
	test_label = new_label_vector[610:720]
	neigh = KNeighborsClassifier(n_neighbors=3)
	neigh.fit(train,train_label)
	predict = neigh.predict(test)
	accuracy = neigh.score(test,test_label)
	print(neigh.score(test,test_label))
	print(predict)


	selected_feat = ['pm10','pm25','label']
	show_data = feature_vector[selected_feat]
	data = {'a0':list(),'a1':list(),'a2':list(),'a3':list(),'a4':list()}
	for j in range(show_data.shape[0]):
		value = show_data.iloc[j]
		if 'a'+str(int(value['label'])) not in data:
			data['a'+str(int(value['label']))]=list()
			temp = {'x': value['pm25'], 'y':value['pm10'] }
			data['a'+str(int(value['label']))].append(temp)
		else:
			temp = {'x': value['pm25'], 'y':value['pm10'] }
			data['a'+str(int(value['label']))].append(temp)
	print(data)
	stats= list()
	stats.append({'y':len(data['a0']), 'label':'unhealthy-high'})
	stats.append({'y':len(data['a1']), 'label':'unhealthy-low'})
	stats.append({'y':len(data['a2']), 'label':'unhealthy'})
	stats.append({'y':len(data['a3']), 'label':'moderate'})
	stats.append({'y':len(data['a4']), 'label':'good'})

	trainstats=610
	teststats = 110
	total = show_data.shape[0]
	classifier = "K Nearest Neighbour"

	
	aqi_vector = feature_vector[['pm25','co','index','label']]

	all_aqi = list()
	for l in range(0,720):
		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(value['pm25'])
		temp = {'x':dt, 'y':lvl}
		all_aqi.append(temp)
	
	aqi = list()
	for l in range(0,450):
		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(value['pm25'])
		temp = {'x':dt, 'y':lvl}
		aqi.append(temp)
	temp_aqi = list(aqi)

	index = 450
	aqi_forecast = list()
	for k in range(0,170):
		sum=0
		for l in range(index-21,index):
			sum+=temp_aqi[l]['y']
		sum=sum/21
		t = {'x':all_aqi[index-1]['x'],'y':sum}
		aqi_forecast.append(t)
		temp_aqi.append(t)	
		index+=1
	
	pollution_rate = 0
	air_pollution_rate = list()
	for l in range(1,721):
		pollution_rate += float(aqi_vector.iloc[l]['pm25'])-float(aqi_vector.iloc[l-1]['pm25'])

		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(aqi_vector.iloc[l]['pm25'])-float(aqi_vector.iloc[l-1]['pm25'])
		temp = {'x':dt, 'y':lvl}
		air_pollution_rate.append(temp)
	pollution_rate = pollution_rate/721

	air_pollution_rate_his = list()
	for l in range(1,450):
		pollution_rate += float(aqi_vector.iloc[l]['pm25'])-float(aqi_vector.iloc[l-1]['pm25'])

		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(aqi_vector.iloc[l]['pm25'])-float(aqi_vector.iloc[l-1]['pm25'])
		temp = {'x':dt, 'y':lvl}
		air_pollution_rate_his.append(temp)
	
	temp_air_pol_rate = list(air_pollution_rate_his)
	index = 447
	air_pollution_rate_for = list()
	for k in range(0,170):
		sum=0
		for l in range(index-21,index):
			sum+=temp_air_pol_rate[l]['y']
		sum=sum/21
		t = {'x':air_pollution_rate[index-1]['x'],'y':sum}
		air_pollution_rate_for.append(t)
		temp_air_pol_rate.append(t)	
		index+=1
	


	all_heat_rate = list()
	for l in range(0,720):
		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(value['co'])
		temp = {'x':dt, 'y':lvl}
		all_heat_rate.append(temp)
	
	heat_rate = list()
	for l in range(0,450):
		value = aqi_vector.iloc[l]
		dt = value['index']
		lvl = float(value['co'])
		temp = {'x':dt, 'y':lvl}
		heat_rate.append(temp)
	temp_heat_rate = list(heat_rate)

	index = 450
	heat_rate_forecast = list()
	for k in range(0,170):
		sum=0
		for l in range(index-21,index):
			sum+=temp_heat_rate[l]['y']
		sum=sum/21
		t = {'x':all_heat_rate[index-1]['x'],'y':sum}
		heat_rate_forecast.append(t)
		temp_heat_rate.append(t)	
		index+=1
	
	return render(request, 'weather.html',{"data":data, 'stat':stats, 'total':total, 'heat_rate_forecast':heat_rate_forecast,'heat_rate':heat_rate ,'air_pollution_rate_for':air_pollution_rate_for,'air_pollution_rate_his':air_pollution_rate_his,'pollution_rate':pollution_rate,'aqi':aqi,'aqi_forecast':aqi_forecast, 'train':trainstats, 'test':teststats, 'classifier':classifier })




def energy(request):
	direc = os.getcwd()+"\classification\Electricity_Data.csv"
	print(direc)
	df = pd.read_csv(direc)
	# feature_column = ['Population','Transmitted energy GWH','Net Electricity Consumption Kwh','Lost Electricty Kwh']
	# label_column = ['Labels']
	# feature_vector = df[feature_column]
	# label_vector = df[label_column]
	# label_vector['Labels'] = label_vector['Labels'].map({'Poor': 0, 'Low': 1, 'Average':2, 'High':3})
	# train  = feature_vector[0:300]
	# test = feature_vector[300:420]
	# train_label = label_vector[0:300]
	# test_label = label_vector[300:420]
	# neigh = KNeighborsClassifier(n_neighbors=3)
	# neigh.fit(train,train_label)
	# predict = neigh.predict(test)
	# accuracy = neigh.score(test,test_label)
	# print(predict)

	selected_feat = ['Net Electricity Consumption Kwh','Gross Production GWH']
	show_data = df[selected_feat]
	data = list()
	for j in range(show_data.shape[0]):
		value = show_data.iloc[j]
		temp = {'x': value['Net Electricity Consumption Kwh'], 'y':int(value['Gross Production GWH'])*1000000 }
		data.append(temp)
	print(data)
	# stats= list()
	# stats.append({'y':len(data['Poor']), 'label':'Poor'})
	# stats.append({'y':len(data['Low']), 'label':'Low'})
	# stats.append({'y':len(data['Average']), 'label':'Average'})
	# stats.append({'y':len(data['High']), 'label':'High'})
	# trainstats=300
	# teststats = 120
	# total = show_data.shape[0]
	# classifier = "K Nearest Neighbour"

	efficiency_vector = df[['Date (Montly)','Gross Production GWH','Net Electricity Consumption Kwh']]
	effic = list()
	for k in range(efficiency_vector.shape[0]):
		value = efficiency_vector.iloc[k]
		dt = value['Date (Montly)']
		eff = float(int(value['Gross Production GWH'])*1000000)-float(value['Net Electricity Consumption Kwh'])
		#timestamp=datetime.datetime(int(dt.split('-')[0]), int(dt.split('-')[1]), int(dt.split('-')[2].split(' ')[0]))
		timestamp = dt
		temp = {'x':timestamp, 'y':eff}
		effic.append(temp)
	
	
	saving = list()
	for l in range(0,380):
		value = efficiency_vector.iloc[l]
		dt = value['Date (Montly)']
		save = float(int(value['Gross Production GWH'])*1000000)-float(value['Net Electricity Consumption Kwh'])
		temp = {'x':dt, 'y':save}
		saving.append(temp)
	temp_saving = list(saving)
	index = 380
	saving_forecast = list()
	for k  in range(0,40):
		sum=0
		for l in range(index-21,index):
			sum+=temp_saving[l]['y']
		sum=sum/21
		t = {'x':effic[index-1]['x'],'y':sum}
		saving_forecast.append(t)
		temp_saving.append(t)	
		index+=1

	

	return render(request, 'energy.html',{"data":data, 'saving':saving,'saving_forecast':saving_forecast })

	

	
	

	


	

	


	







