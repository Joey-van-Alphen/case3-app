#!/usr/bin/env python
# coding: utf-8

# # Case 3 - Team 12

# * Joey van Alphen
# * Mohamed Garad
# * Nusret Kaya
# * Shereen Macnack

# In[1]:


#pip install streamlit-folium
#!pip install streamlit
#!pip install statsmodels
#!pip install cbsodata


# In[2]:


import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import folium_static
import numpy as np
import geopandas as gpd
import plotly.express as px
import statsmodels.api as sm
import cbsodata
import matplotlib.pyplot as plt
from folium.plugins import MarkerCluster
import plotly.graph_objects as go


# ## Dataset 1 Open Charge Map

# In[3]:


response = requests.get('https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&compact=true&verbose=false&key=6ba1f76e-aefd-4fca-aeea-caa80b9e24a3')
json = response.json()
df = pd.DataFrame(json)

df.head()


# In[4]:


response = requests.get('https://api.openchargemap.io/v3/poi/?output=json&countrycode=NL&maxresults=1000&compact=true&verbose=false&key=6ba1f76e-aefd-4fca-aeea-caa80b9e24a3')
responsejson = response.json()
###Dataframe bevat kolom die een list zijn.
#Met json_normalize zet je de eerste kolom om naar losse kolommen
Laadpalen = pd.json_normalize(responsejson)
#Daarna nog handmatig kijken welke kolommen over zijn in dit geval Connections
#Kijken naar eerst laadpaal op de locatie
#Kan je uitpakken middels:
df4 = pd.json_normalize(Laadpalen.Connections)
df5 = pd.json_normalize(df4[0])
df5.head()
###Bestanden samenvoegen
Laadpalen = pd.concat([Laadpalen, df5], axis=1)
Laadpalen.head()


# In[5]:


#Laadpalen.columns


# In[6]:


df2 = (Laadpalen[['AddressInfo.ID', 'AddressInfo.Title', 'AddressInfo.AddressLine1',
       'AddressInfo.Town', 'AddressInfo.StateOrProvince',
       'AddressInfo.Postcode', 'AddressInfo.CountryID','AddressInfo.Latitude', 'AddressInfo.Longitude', 'ConnectionTypeID', 'Quantity']])
df2.head()


# In[7]:


#df2['AddressInfo.StateOrProvince'].unique()


# In[8]:


df2['AddressInfo.StateOrProvince'].replace(["North Holland", "Holandia Północna","Nordholland", 'North-Holland', 'Noord Holand', 'NH'], 'Noord-Holland', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['South Holland', 'Zuid Holland', 'Zuid-Holland ', 'ZH', 'Stellendam' ], 'Zuid-Holland', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['Seeland'], 'Zeeland', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['Stadsregio Arnhem Nijmegen'], 'Gelderland', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['UTRECHT', 'UT'], 'Utrecht', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['Noord Brabant', 'North Brabant'], 'Noord-Brabant', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['FRL'], 'Friesland', inplace = True)
df2['AddressInfo.StateOrProvince'].replace(['Noord Brabant', 'North Brabant'], 'Noord-Brabant', inplace = True)
df2['AddressInfo.StateOrProvince'].replace([''], 'Missing', inplace = True)


# In[9]:


df2['AddressInfo.StateOrProvince'].unique()


# In[10]:


aantal_per_prov = df2['AddressInfo.StateOrProvince'].value_counts()
df_prov = pd.DataFrame(aantal_per_prov)
df_prov = df_prov.reset_index(level=0)
df_prov.columns = ['provincie','aantal laadpalen']
df_prov.drop(df_prov[df_prov['provincie']=='Missing'].index, inplace = True)
df_prov.head(15)


# In[11]:


fig1 = px.bar(df_prov, x='provincie', y='aantal laadpalen', title = 'Aantal laadpalen per provincie')
fig1.update_xaxes(title_text="Provincie")
fig1.update_yaxes(title_text="Aantal laadpalen")
#fig.show()



# In[12]:


fig2 = px.pie(df_prov, values='aantal laadpalen', names='provincie', title='Aantal laadpalen per provincie')
#fig.show()


# ### Dataset met geometries combineren

# In[13]:


#countries = gpd.read_file('Grenzen_van_alle_Nederlandse_gemeenten_en_provincies.geojson')
#countries.head()
geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json'
gemeentegrenzen = gpd.read_file(geodata_url)
gemeentegrenzen.head()


# In[ ]:





# In[14]:


aantal_per_gem = df2['AddressInfo.Town'].value_counts()
df4 = pd.DataFrame(aantal_per_gem)
df5 = df4.reset_index(level=0)
df5.columns = ['statnaam','aantal_laadpalen']
array = df5['statnaam'].unique()
np.sort(array)


# In[15]:


df5.replace(["s-Hertogenbosch", "'s Hertogenbosch"], 's-Hertogenbosch', inplace = True)
df5.replace(['Utrecht '], 'Utrecht', inplace = True)
df5.replace(['Den Haag'], 'Den haag', inplace = True)


# In[ ]:





# In[16]:


df_samen = df5.merge(gemeentegrenzen, how='left', on='statnaam')
df_samen = df_samen[['statnaam', 'aantal_laadpalen', 'geometry']]
df_samen.head()


# In[17]:


df_samen.isna().sum()


# In[18]:


df_samen.dropna(inplace=True)


# In[19]:


df_samen.isna().sum()


# In[20]:


geo_df_crs = {'init': 'epsg:4326'}
geo_df = gpd.GeoDataFrame(df_samen, crs= geo_df_crs, geometry = df_samen.geometry)
geo_df.head()


# In[21]:


location_select = st.sidebar.selectbox('Welk gebied wil je zien?', ('Noord-Holland', 'Zuid-Holland', 'Zeeland', 'Noord-Braant', 'Utrecht', 'Flevoland', 'Friesland', 'Groningen','Drenthe', 'Overijssel', 'Gelderland', 'Limburg'))

if location_select == 'Noord-Holland': 
    location = [52.520587, 4.788474]
elif location_select =='Zuid-Holland':
    location = [52.020798, 4.493784]
elif location_select == 'Zeeland':
    location = [51.4940309, 3.8496815]
elif location_select =='Noord-Brabant':
    location = [51.482654, 5.232169]
elif location_select == 'Utrecht':
    location = [52.0907374, 5.1214201]
elif location_select =='Flevoland':
    location = [52.527978, 5.595351]
elif location_select == 'Friesland':
    location = [53.1641642, 5.7817542]
elif location_select =='Groningen':
    location = [53.2193835, 6.5665018]
elif location_select == 'Drenthe':
    location = [52.947601, 6.623059]
elif location_select =='Overijssel':
    location = [52.438781, 6.501641]
elif location_select == 'Gelderland':
    location = [52.045155, 5.871823]
elif location_select == 'Limburg':
    location = [50.398601, 8.079578]


# In[22]:


m = folium.Map(location= [52.371807, 4.896029], zoom_start = 7)


# In[23]:


m = folium.Map(location=location, zoom_start=10)

m.choropleth(
    geo_data=geo_df,
    name="geometry",
    data=geo_df,
    columns= ["statnaam","aantal_laadpalen"],
    key_on="feature.properties.statnaam",
    fill_color="RdYlGn_r",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Aantal laadpalen per gemeente")

folium.features.GeoJson(geo_df,  
                        name='Labels',
                        style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                        tooltip=folium.features.GeoJsonTooltip(fields=['aantal_laadpalen'],
                                                                aliases = ['Aantal laadpalen:'],
                                                                labels=True,
                                                                sticky=False
                                                                            )
                       ).add_to(m)



# In[24]:


m2 = folium.Map(location=location, zoom_start=10)

marker_cluster = MarkerCluster().add_to(m2)

for row in df2.iterrows():
    row_values = row[1]
    location = [row_values['AddressInfo.Latitude'], row_values['AddressInfo.Longitude']]
    popup = popup = '<strong>' + row_values['AddressInfo.Title'] + '</strong>'
    marker = folium.Marker(location = location, popup = popup)
    marker.add_to(marker_cluster)



# In[25]:


#duplicated = df.duplicated(subset = 'AddressInfo.Town', keep = False)


# ## Dataset 2: laadpaaldata

# In[26]:


df = pd.read_csv('laadpaaldata.csv')
df.head(100)


# In[27]:


df.info()


# In[28]:


df.isna().sum()


# In[29]:


#df.loc[df.duplicated(subset='Started', keep = False)]
#beide 'duplicates' hebben andere eindtijd en andere waardes, dus hier is geen sprake van duplicates


# In[30]:


df['ConnectedTime'].hist() 



# In[31]:


#outliers verwijderen
df2 = df[df['ConnectedTime'] <= 30] 
df2['ConnectedTime'].hist()


# In[32]:


df2['ChargeTime'].hist() 


# In[33]:


df3 = df2[df2['ChargeTime']>0] 
df4 = df3[df3['ChargeTime']<9]
df4['ChargeTime'].hist()


# In[34]:


df4['bezethouden'] = df4['ConnectedTime']-df4['ChargeTime']
df5 = df4[df4['bezethouden']>0]


# In[35]:


df5['bezethouden'].hist()


# In[36]:


#df5.loc[df5['ChargeTime'] <=0]


# In[37]:


fig3 = px.histogram(df5, x="bezethouden", nbins=25, title='Bezethouden van een laadpaal')

mean_bezethouden = df5['bezethouden'].mean()
median_bezethouden = df5['bezethouden'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_bezethouden}"} 
my_annotation2 = {'x': 0.9, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_bezethouden}"} 
fig3.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig3.update_xaxes(title_text="Het bezethouden van een laadpaal in uren")
fig3.update_yaxes(title_text="Aantal")

#fig3.show()


# In[38]:


fig4 = px.histogram(df5, x="ChargeTime", nbins= 20, title='Laadtijd van een laadpaal')

mean_chargetime = df5['ChargeTime'].mean()
median_chargetime = df5['ChargeTime'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_chargetime}"} 
my_annotation2 = {'x': 0.8, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_chargetime}"} 
fig4.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig4.update_xaxes(title_text="De tijd dat de laadpaal echt aan het laden is in uren")
fig4.update_yaxes(title_text="Aantal")

#fig4.show()


# In[39]:


fig5 = px.histogram(df5, x="ConnectedTime", nbins=25, title='Tijd verbonden aan een laadpaal')

mean_connectedtime = df5['ConnectedTime'].mean()
median_connectedtime = df5['ConnectedTime'].median()
my_annotation1 = {'x': 0.1, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"Het gemiddelde is: {mean_connectedtime}"} 
my_annotation2 = {'x': 0.9, 'y': 1.05, 'xref': 'paper', 'yref': 'paper', 'showarrow': False,"text": f"De mediaan is: {median_connectedtime}"} 
fig5.update_layout({'annotations': [my_annotation1, my_annotation2]})

fig5.update_xaxes(title_text="De tijd dat de laadpaal is verbonden in uren")
fig5.update_yaxes(title_text="Aantal")

#fig5.show()


# In[40]:


fig6 = px.histogram(df5, x=['bezethouden', 'ChargeTime'], nbins=25, barmode="overlay", title='Het bezethouden tegen de daadwerkelijke charge time')
fig6.update_xaxes(title_text="Tijd in uren")
fig6.update_yaxes(title_text="Aantal")
#fig6.show()


# In[41]:


data_select = st.sidebar.selectbox('Welke maand wil je zien?', ('Januari', 'Februari', 'Maart', 'April', 'Mei', 'Juni', 'Juli', 'Augustus', 'September', 'Oktober', 'November', 'December'))

if data_select == 'Januari': 
    data = df5[(df5['Ended']>='2018-01-01')&(df5['Ended']<='2018-01-31')].sort_values(by='Ended')
elif data_select =='Februari':
    data = df5[(df5['Ended']>='2018-02-01')&(df5['Ended']<='2018-02-31')].sort_values(by='Ended')
elif data_select == 'Maart':
    data = df5[(df5['Ended']>='2018-03-01')&(df5['Ended']<='2018-03-31')].sort_values(by='Ended')
elif data_select =='April':
    data = df5[(df5['Ended']>='2018-04-01')&(df5['Ended']<='2018-04-31')].sort_values(by='Ended')
elif data_select == 'Mei':
    data = df5[(df5['Ended']>='2018-05-01')&(df5['Ended']<='2018-05-31')].sort_values(by='Ended')
elif data_select =='Juni':
    data = df5[(df5['Ended']>='2018-06-01')&(df5['Ended']<='2018-06-31')].sort_values(by='Ended')
elif data_select == 'Juli':
    data = df5[(df5['Ended']>='2018-07-01')&(df5['Ended']<='2018-07-31')].sort_values(by='Ended')
elif data_select == 'Augustus':
    data = df5[(df5['Ended']>='2018-08-01')&(df5['Ended']<='2018-08-31')].sort_values(by='Ended')
elif data_select =='September':
    data = df5[(df5['Ended']>='2018-09-01')&(df5['Ended']<='2018-09-31')].sort_values(by='Ended')
elif data_select == 'Oktober':
    data = df5[(df5['Ended']>='2018-10-01')&(df5['Ended']<='2018-10-31')].sort_values(by='Ended')
elif data_select =='November':
    data = df5[(df5['Ended']>='2018-11-01')&(df5['Ended']<='2018-11-31')].sort_values(by='Ended')
elif data_select == 'December':
    data = df5[(df5['Ended']>='2018-12-01')&(df5['Ended']<='2018-12-31')].sort_values(by='Ended')
    
    
fig7 = px.line(data, x='Ended', y='TotalEnergy', title = 'Laadprofiel per maand')
fig7.update_xaxes(title_text="Datum")
fig7.update_yaxes(title_text="Totaal verbruikte energie in Wh")
#fig7.show()


# In[42]:


fig8 = px.scatter(df5, x='TotalEnergy', y ='MaxPower', trendline="ols", title = 'Scatterplot van de totale energie verbruik tegenover het maximaal gevraagde vermogen')
fig8.update_xaxes(title_text="Totaal verbruikte energie in Wh")
fig8.update_yaxes(title_text="Maximaal gevraagde vermogen in W")


# ## Dataset 3: Elektrische voertuigen

# Voor het publiceren van de app een restrictie in file grootte, daarom eerst gefilterde file exporteren en opnieuw gebruiken

# In[43]:


#df1 = pd.read_csv('Elektrische_voertuigen.csv', low_memory=False)
#df1.head()


# In[44]:


#df1.columns


# In[45]:


#elektrische_voertuigen = df1[['Merk', 'Voertuigsoort','Handelsbenaming', 'Catalogusprijs', 'Datum eerste toelating', 'Cilinderinhoud']]
#elektrische_voertuigen_streamlit  = elektrische_voertuigen.assign(Datum_eerste_toelating = pd.to_datetime(elektrische_voertuigen['Datum eerste toelating'], format='%Y%m%d') )
#elektrische_voertuigen_streamlit.head()


# In[46]:


#elektrische_voertuigen_streamlit.to_csv(r"C:\HvA 2022-2023\Minor Data Science\Case3\elektrische_voertuigen_streamlit.csv", index = False)

elektrische_voertuigen = pd.read_csv('elektrische_voertuigen_streamlit.csv')
elektrische_voertuigen.head()


# In[47]:


elektrische_voertuigen = elektrische_voertuigen.assign(Type = np.where(elektrische_voertuigen['Cilinderinhoud'].isna(),'Electric','Hybrid'))
elektrische_voertuigen.head()


# In[48]:


#elektrische_voertuigen.loc[elektrische_voertuigen['Type'] == 'Hybrid']


# In[49]:


elektrische_voertuigen['Merk'].unique()


# In[50]:


elektrische_voertuigen['Merk'] = elektrische_voertuigen['Merk'].replace(['TESLA MOTORS', 'BMW I', 'VW', 'FORD-CNG-TECHNIK', 'VOLKSWAGEN/ZIMNY', 'JAGUAR CARS', 'M.A.N.'], 
                                                      ['TESLA', 'BMW', 'VOLKSWAGEN', 'FORD', 'VOLKSWAGEN', 'JAGUAR', 'MAN'])


# In[51]:


array = elektrische_voertuigen['Merk'].unique()
np.sort(array)


# In[52]:


aantal_auto = elektrische_voertuigen['Merk'].value_counts()
df = pd.DataFrame(aantal_auto)
df = df.reset_index(level=0)
df.columns = ['merk','aantal']
df.head()


# In[53]:


fig9 = px.bar(df, x='merk', y='aantal', title='Top 20 verkochte automerken')
fig9.update_layout(xaxis_range=[0,20])
fig9.update_xaxes(title_text="Merk")
fig9.update_yaxes(title_text="Aantal")
#fig9.show()


# In[54]:


data_t = elektrische_voertuigen[elektrische_voertuigen['Merk'] == 'TESLA']
data_t['Handelsbenaming'].value_counts()


# In[55]:


data_t.replace(['TESLA MODEL 3', 'MODEL3', 'Model 3', 'VIO EUROPE MODEL 3'], 'MODEL 3', inplace = True)
data_t.replace(['TESLA MODEL S', 'MODEL S 70', 'S 75 D', 'TESLA MODEL S 75 D', 'MODEL S 85', 'MODEL S P85+'], 'MODEL S', inplace = True)
data_t.replace(['TESLA MODEL X'], 'MODEL X', inplace = True)
data_t.replace(['TESLA MODEL Y', 'Model Y'], 'MODEL 3', inplace = True)
tesla_models = pd.DataFrame(data_t['Handelsbenaming'].value_counts())
tesla_models = tesla_models.reset_index(level=0)
tesla_models.columns = ['model','aantal']


# In[56]:


fig10 = px.pie(tesla_models, values='aantal', names='model', title ='Verkochte modellen van de Tesla')

#fig10.show()


# In[57]:


df_per_type = elektrische_voertuigen[['Datum_eerste_toelating', 'Type']]
df_per_type.sort_values(by='Datum_eerste_toelating')
df_per_type.head()


# In[58]:


#fig = px.line(df_per_type, x='Datum_eerste_toelating', y=df_per_type['Datum_eerste_toelating'].value_counts())
#fig.show()


# In[59]:


df_electric = df_per_type[df_per_type['Type']=='Electric']
#aantal_eper_jaar = df_electric['Datum_eerste_toelating'].value_counts()
#df_e = pd.DataFrame(aantal_eper_jaar)
#df_e = df.reset_index(level=0)
#df_e.head()
el = df_electric.value_counts('Datum_eerste_toelating')
el_df = pd.DataFrame(el).sort_values(by='Datum_eerste_toelating')
el_df = el_df.reset_index(level=0)
el_df = el_df[el_df['Datum_eerste_toelating']>='2000-01-01']
el_df['cumsum'] = el_df[0].cumsum()
el_df.head()


# In[60]:


df_hybrid = df_per_type[df_per_type['Type']=='Hybrid']
#aantal_hper_jaar = df_hybrid['Datum_eerste_toelating'].value_counts()
#df2 = pd.DataFrame(aantal_hper_jaar)
#df2 = df.reset_index(level=0)
#df2.head()
hy = df_hybrid.value_counts('Datum_eerste_toelating')

hy_df = pd.DataFrame(hy).sort_values(by='Datum_eerste_toelating')
hy_df = hy_df.reset_index(level=0)
hy_df = hy_df[hy_df['Datum_eerste_toelating']>='2000-01-01']
hy_df['cumsum'] = hy_df[0].cumsum()
hy_df.head()


# In[62]:


fig11 = go. Figure()
fig11.add_trace(go.Scatter(x=hy_df['Datum_eerste_toelating'], y=hy_df['cumsum']))
fig11.add_trace(go.Scatter(x=el_df['Datum_eerste_toelating'], y=el_df['cumsum']))
fig11.update_xaxes(title_text="Jaar")
fig11.update_yaxes(title_text="Aantal verkochte auto's (cumulatief)")
fig11.update_layout(title_text="Aantal verkochte auto's per type") 
#fig11.show()


# In[63]:


aantal_per_jaar = elektrische_voertuigen['Datum_eerste_toelating'].value_counts()
df = pd.DataFrame(aantal_per_jaar)
df = df.reset_index(level=0)
df= df[df['index']>'2010']
df = df.sort_values(by= 'index')
df['cumsum'] = df['Datum_eerste_toelating'].cumsum()
df.head()

#df['index'] = df['index'].dt.year
#df = df.groupby("index")['Datum_eerste_toelating'].count()
#df.head(100)


# In[64]:


fig12 = px.line(df, x='index', y ='cumsum', title="Totaal aantal verkochte auto's" )
fig12.update_xaxes(title_text="Jaar")
fig12.update_yaxes(title_text="Aantal verkochte auto's (cumulatief)")
#fig12.show()


# In[68]:


df_afjaar = df[(df['index']>'2021-10-1')&(df['index']<'2022-10-1')]
df_afjaar.head()
fig13 = px.line(df_afjaar, x='index', y ='cumsum', title= "Aantal verkochte elektrische auto's in het afegelopen jaar")
fig13.update_xaxes(title_text="Jaar")
fig13.update_yaxes(title_text="Aantal verkochte auto's (cumulatief)")
#fig13.show()


# ## Streamlit deel

# In[69]:


st.title('Case 3 Elektrisch vervoer')
st.markdown("Dashboard van 3 datasets: Open Charge Map, Laadpaaldata en Elektrische voertuigen")


# In[70]:


st.header("1. Open Charge Map")
st.markdown('Het aantal laadpunten per gebied')
folium_static(m)
folium_static(m2)
st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.header("2. Laadpaaldata")
st.plotly_chart(fig4)
st.plotly_chart(fig5)
st.plotly_chart(fig3)
st.plotly_chart(fig6)
st.plotly_chart(fig7)
st.plotly_chart(fig8)

st.header("3. Elektrische voertuigen ")
st.plotly_chart(fig9)
st.plotly_chart(fig10)
st.plotly_chart(fig11)
st.plotly_chart(fig12)
st.plotly_chart(fig13)

