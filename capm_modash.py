import streamlit as st 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.api as sm
import numpy as np
import io
import plotly.express as px
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import GLSAR


st.set_page_config(layout="wide")

st.markdown("""
    <h1 style="text-align: center;">LMIC DASHBOARD</h1>
""", unsafe_allow_html=True) # Título

# Importar bases 
population = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=8)
lmic = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 9)
mark_cap = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx' , sheet_name= 7)
# returns = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name = 3 )
famafrench = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  10)
msci = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  11)
prices = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  1)
rate_ex = pd.read_excel(r'C:\Users\gadyh\OneDrive\Documentos\UNISABANA\CAPM - WORLDBANK\list_of_countries_index.xlsx', sheet_name=  5)


mark_cap.set_index('Country', inplace = True)
mark_cap = mark_cap.iloc[:, 0:1]
prices.set_index('Country', inplace = True)
rate_ex.set_index('Country', inplace = True)
population.set_index('Country', inplace=True)

# Calcular retornos en USD
usd_price = prices / rate_ex
returns = usd_price.pct_change(axis = 1)

# Eliminar paises vacios
returns = returns.iloc[:, 1:].dropna(how = 'all')
mark_cap = mark_cap.dropna(how = 'all')
population = population.iloc[:, :2]

# Filtrar df por paises en comun
# Encontrar los índices en común entre los tres DataFrames
common_countries = returns.index.intersection(mark_cap.index).intersection(population.index)
mark_cap = mark_cap.loc[common_countries]
returns = returns.loc[common_countries]
population = population.loc[common_countries]

# Seleccion de rango
min_pop = int(population['Population'].min())
max_pop = int(population['Population'].max())

# Entrada de rango
st.subheader('Choose range for groups of countries: ')

option = st.radio("segmentation method :", ["Terciles", "Quartiles", "Customized"])
if option == "Terciles":
    percentiles = np.percentile(population['Population'], [25, 50, 75])
    group1_max, group2_max, group3_max = percentiles
elif option == "Quartiles":
    percentiles = np.percentile(population['Population'], [20, 40, 60, 80])
    group1_max, group2_max, group3_max = percentiles[:3]
else:
    # 📌 OPCIÓN 2: Entrada manual
    st.subheader("Choose the range: ")
    group1_max = st.number_input("Upper range for group 1:", min_value=0, value = 3_000_000)
    group2_max = st.number_input("Upper range for group 2:", min_value=group1_max, value= 10_000_000)
    group3_max = st.number_input("Upper range for group 3:", min_value=group2_max, value= 30_000_000)
    
# 📌 Definir los grupos basados en la opción seleccionada
group1 = population[population["Population"] <= group1_max]
group2 = population[(population["Population"] > group1_max) & (population["Population"] <= group2_max)]
group3 = population[(population["Population"] > group2_max)  & (population["Population"] <= group3_max)]
group4 = population[(population["Population"] > group3_max)]

# 📌 Función para asignar grupos según la población
def assign_group(population):
    if population < group1_max:
        return "Group 1: Very Low"
    elif population < group2_max:
        return "Group 2: Low"
    elif population < group3_max:
        return "Group 3: Medium"
    else:
        return "Group 4: High"

# Asignar grupos
population["Group"] = population["Population"].apply(assign_group)

# 📌 Definir colores para los 4 grupos
color_map = {
    "Group 1: Very Low": "lightblue",
    "Group 2: Low": "green",
    "Group 3: Medium": "orange",
    "Group 4: High": "red"
}

# 📌 Crear el mapa con Plotly
fig = px.choropleth(
    population, 
    locations="Suffix",  # Código de país ISO 3
    color="Group",
    title="Distribution of countries by population: ",
    color_discrete_map=color_map,
    projection="natural earth"
)

fig.update_layout(height=700, width=1200)
st.plotly_chart(fig)    

# Paises por grupo
col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.write('Countries of group 1: ')
    st.dataframe(group1, use_container_width=True)
    st.write('Number of countries in group 1: ', group1['Suffix'].count())

with col2: 
    st.write('Countries of group 2: ')
    st.dataframe(group2, use_container_width=True)
    st.write('Number of countries in group 2: ', group2['Suffix'].count())
    
with col3: 
    st.write('Countries of group 3: ')
    st.dataframe(group3, use_container_width=True) 
    st.write('Number of countries in group 3: ', group3['Suffix'].count())

with col4: 
    st.write('Countries of group 4: ')
    st.dataframe(group4, use_container_width=True) 
    st.write('Number of countries in group 4: ', group4['Suffix'].count())
    
# Winsor2
def winsorize_series(series, lower_percentile=1, upper_percentile=99):
    lower = np.percentile(series, lower_percentile)
    upper = np.percentile(series, upper_percentile)
    return np.clip(series, lower, upper)


returns = returns.apply(lambda row: winsorize_series(row), axis=1) 

# Filtrar retornos y market cap por grupo
return_1 = returns[returns.index.isin(group1.index)]
return_2 = returns[returns.index.isin(group2.index)]
return_3 = returns[returns.index.isin(group3.index)]
return_4 = returns[returns.index.isin(group4.index)]

mark_cap1 = mark_cap[mark_cap.index.isin(group1.index)]
mark_cap2 = mark_cap[mark_cap.index.isin(group2.index)]
mark_cap3 = mark_cap[mark_cap.index.isin(group3.index)]
mark_cap4 = mark_cap[mark_cap.index.isin(group4.index)]

# Media de los retornos
return_1_mean = return_1.mean()
return_2_mean = return_2.mean()
return_3_mean = return_3.mean()
return_4_mean = return_4.mean()

return_1_mean = return_1_mean.to_frame(name = 'Returns')
return_2_mean = return_2_mean.to_frame(name = 'Returns')
return_3_mean = return_3_mean.to_frame(name = 'Returns')
return_4_mean = return_4_mean.to_frame(name = 'Returns')

# Tratar famafrench
famafrench['RF'] = pd.to_numeric(famafrench['RF'], errors='coerce')
famafrench['Mkt-RF'] = pd.to_numeric(famafrench['Mkt-RF'], errors='coerce')# Volver a numero
famafrench.set_index('date', inplace=True)

# Ri - RF
return1_rf = (return_1_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
return2_rf = (return_2_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
return3_rf = (return_3_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')
return4_rf = (return_4_mean['Returns'] - famafrench['RF']).to_frame(name = 'Ri - Rf')


# CAPM Simple non weighted

st.markdown("""
    <h1 style="text-align: center;">SIMPLE MODEL NON WEIGHTED</h1>
""", unsafe_allow_html=True) 

def extract_model_summary(model):
    """Extrae coeficientes, errores estándar, p-valores, R² y otras métricas."""
    summary_df = pd.DataFrame({
        "Coefficients": model.params,
        "Standard Errors": model.bse,
        "p-values": model.pvalues,
        "Lower range 95%": model.conf_int()[0],
        "Upper range 95%": model.conf_int()[1]
    })
    summary_df.loc["R²", "Coefficients"] = model.rsquared
    summary_df.loc["R² Adjusted", "Coefficients"] = model.rsquared_adj
    summary_df.loc["F-Statistical", "Coefficients"] = model.fvalue
    return summary_df

# Grupo 1
Y_1 = return1_rf.iloc[:-1, :]
X = famafrench['Mkt-RF']
X = sm.add_constant(X)  
modelo1 = sm.OLS(Y_1, X).fit()

# Grupo 2
Y_2 = return2_rf.iloc[:-1, :]
modelo2 = sm.OLS(Y_2, X).fit()

# Grupo 3
Y_3 = return3_rf.iloc[:-1, :]
modelo3 = sm.OLS(Y_3, X).fit()

# Grupo 4
Y_4 = return4_rf.iloc[:-1, :]
modelo4 = sm.OLS(Y_4, X).fit()

col1, col2, col3, col4 = st.columns(4)

with col1: 
    st.write('Simple model non weighted for group 1:' )
    modelo1 = extract_model_summary(modelo1)
    st.dataframe(modelo1, use_container_width=True)
    
with col2: 
    st.write('Simple model non weighted for group 2:' )
    modelo2 = extract_model_summary(modelo2)
    st.dataframe(modelo2, use_container_width=True)
    
with col3:
    st.write('Simple model non weighted for group 3:' )
    modelo3 = extract_model_summary(modelo3)
    st.dataframe(modelo3, use_container_width=True) 
    
with col4: 
    st.write('Simple model non weighted for group 4: ')
    modelo4 = extract_model_summary(modelo4)
    st.dataframe(modelo4, use_container_width=True)

# Simple models weighted

st.markdown("""
    <h1 style="text-align: center;">SIMPLE MODEL WEIGHTED</h1>
""", unsafe_allow_html=True) 

famafrench = famafrench.reset_index()

# Grupo 1
return_1 = return_1.fillna(0)
df_long = return_1.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
df_long.rename(columns={"index": "Date"}, inplace=True)
df_final = df_long.merge(mark_cap1, on="Country", how="left")
# Suponiendo que tu DataFrame se llama df
df = df_final.copy()  # Para evitar advertencias en caso de modificar el DataFrame original
# Filtrar para ignorar retornos 0
df_filtered = df[df["Returns"] != 0].copy()
df_filtered["Date"] = pd.to_datetime(df_filtered["Date"])
# Calcular la suma de mark_cap por fecha ignorando retornos 0
sum_mark_cap = df_filtered.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
# Merge con el DataFrame original (incluyendo los valores filtrados)
df_filtered = df_filtered.merge(sum_mark_cap, on="Date", how="left")
# Calcular retorno ponderado
df_filtered["Weighted_Returns"] = df_filtered["Returns"] * (df_filtered["mark_cap"] / df_filtered["sum_mark_cap"])
# Calcular el retorno total ponderado por fecha
weighted1_returns = df_filtered.groupby("Date")["Weighted_Returns"].sum().reset_index()
return4_rf = (weighted1_returns['Weighted_Returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Grupo 2
return_2 = return_2.fillna(0)
df_long1 = return_2.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
df_long1.rename(columns={"index": "Date"}, inplace=True)
df_final1 = df_long1.merge(mark_cap2, on="Country", how="left")
df1 = df_final1.copy()  
df_filtered1 = df1[df1["Returns"] != 0].copy()
df_filtered1["Date"] = pd.to_datetime(df_filtered1["Date"])
sum_mark_cap2 = df_filtered1.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
df_filtered1 = df_filtered1.merge(sum_mark_cap2, on="Date", how="left")
df_filtered1["Weighted_Returns"] = df_filtered1["Returns"] * (df_filtered1["mark_cap"] / df_filtered1["sum_mark_cap"])
weighted2_returns = df_filtered1.groupby("Date")["Weighted_Returns"].sum().reset_index()
return5_rf = (weighted2_returns['Weighted_Returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Grupo 3
return_3 = return_3.fillna(0)
df_long2 = return_3.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
df_long2.rename(columns={"index": "Date"}, inplace=True)
df_final2 = df_long2.merge(mark_cap3, on="Country", how="left")
df2 = df_final2.copy()  
df_filtered2 = df2[df2["Returns"] != 0].copy()
df_filtered2["Date"] = pd.to_datetime(df_filtered2["Date"])
sum_mark_cap3 = df_filtered2.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
df_filtered2 = df_filtered2.merge(sum_mark_cap3, on="Date", how="left")
df_filtered2["Weighted_Returns"] = df_filtered2["Returns"] * (df_filtered2["mark_cap"] / df_filtered2["sum_mark_cap"])
weighted3_returns = df_filtered2.groupby("Date")["Weighted_Returns"].sum().reset_index()
return6_rf = (weighted3_returns['Weighted_Returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Grupo 4
return_4 = return_4.fillna(0)
df_long3 = return_4.reset_index().melt(id_vars="Country", var_name="Date", value_name="Returns")
df_long3.rename(columns={"index": "Date"}, inplace=True)
df_final3 = df_long3.merge(mark_cap4, on="Country", how="left")
df3 = df_final3.copy()  
df_filtered3 = df3[df3["Returns"] != 0].copy()
df_filtered3["Date"] = pd.to_datetime(df_filtered3["Date"])
sum_mark_cap4 = df_filtered3.groupby("Date")["mark_cap"].sum().rename("sum_mark_cap")
df_filtered3 = df_filtered3.merge(sum_mark_cap4, on="Date", how="left")
df_filtered3["Weighted_Returns"] = df_filtered3["Returns"] * (df_filtered3["mark_cap"] / df_filtered3["sum_mark_cap"])
weighted4_returns = df_filtered3.groupby("Date")["Weighted_Returns"].sum().reset_index()
return7_rf = (weighted4_returns['Weighted_Returns'] - famafrench['RF']).to_frame(name = 'WRi - Rf')

# Modelos simples weighted
# modelo 1
Y_4 = return4_rf.iloc[:-1, :]
X = famafrench['Mkt-RF']
X = sm.add_constant(X)  
modelo5 = sm.OLS(Y_4, X).fit()

# Modelo 2
Y_5 = return5_rf.iloc[:-1, :]
modelo6 = sm.OLS(Y_5, X).fit()

# Modelo 3
Y_6 = return6_rf.iloc[:-1, :]
modelo7 = sm.OLS(Y_6, X).fit()

# Modelo 4
Y_7 = return7_rf.iloc[:-1, :]
modelo8 = sm.OLS(Y_7, X).fit()

col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.write('Simple model weighted for group 1:' )
    modelo5 = extract_model_summary(modelo5)
    st.dataframe(modelo5, use_container_width=True) 

with col2: 
    st.write('Simple model weighted for group 2:' )
    modelo6 = extract_model_summary(modelo6)
    st.dataframe(modelo6, use_container_width=True) 

with col3: 
    st.write('Simple model non weighted for group 3:' )
    modelo7 = extract_model_summary(modelo7)
    st.dataframe(modelo7, use_container_width=True) 

with col4: 
    st.write('Simple modelo weighted for group 4:')
    modelo8 = extract_model_summary(modelo8)
    st.dataframe(modelo8, use_container_width=True)
    
   
# Multifactorial model non weighted

st.markdown("""
    <h1 style="text-align: center;">MULTIFACTORIAL MODEL NON WEIGHTED</h1>
""", unsafe_allow_html=True) 

# Grupo 1
Y_7 = return1_rf.reset_index().iloc[:-1, 1:]
X_11 = msci.iloc[:-1, :1]
X_1 = pd.concat([famafrench['Mkt-RF'], X_11], axis = 1)
X_1 = sm.add_constant(X_1)

model9 = sm.OLS(Y_7, X_1).fit()

# grupo 2
Y_8 = return2_rf.reset_index().iloc[:-1, 1:]
X_22 = msci.iloc[:-1, :1]
X_2 = pd.concat([famafrench['Mkt-RF'], X_22], axis = 1)
X_2 = sm.add_constant(X_2)
model10 = sm.OLS(Y_8, X_2).fit()

# Grupo 3
Y_9 = return3_rf.reset_index().iloc[:-1, 1:]
X_33 = msci.iloc[:-1, :1]
X_3 = pd.concat([famafrench['Mkt-RF'], X_33], axis = 1)
X_3 = sm.add_constant(X_3)
model11 = sm.OLS(Y_9, X_3).fit()

# Grupo 4
Y_10 = return4_rf.reset_index().iloc[:-1, 1:]
X_44 = msci.iloc[:-1, :1]
X_4 = pd.concat([famafrench['Mkt-RF'], X_44], axis = 1)
X_4 = sm.add_constant(X_4)
model12 = sm.OLS(Y_10, X_4).fit()


col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.write('Multifactorial model non weighted for group 1:' )
    modelo9 = extract_model_summary(model9)
    st.dataframe(modelo9, use_container_width=True) 

with col2: 
    st.write('Multifactorial model non weighted for group 2:' )
    modelo10 = extract_model_summary(model10)
    st.dataframe(modelo10, use_container_width=True) 

with col3: 
    st.write('Multifactorial model non weighted for group 3:' )
    modelo11 = extract_model_summary(model11)
    st.dataframe(modelo11, use_container_width=True) 
    
with col4: 
    st.write('Multifactorial model non weighted for group 4:')
    modelo12 = extract_model_summary(model12)
    st.dataframe(modelo12, use_container_width=True)
    
    
# Multifactorial model Weighted
st.markdown("""
    <h1 style="text-align: center;">MULTIFACTORIAL MODEL WEIGHTED</h1>
""", unsafe_allow_html=True) 

# Modelo 1
Y_11 = return4_rf.iloc[:-1, :]
X_55 = msci.iloc[:-1, :1]
X_5 = pd.concat([famafrench['Mkt-RF'], X_55], axis = 1)
X_5 = sm.add_constant(X_5)
model10 = sm.OLS(Y_11, X_5).fit()

# Modelo 2
Y_12 = return5_rf.iloc[:-1, :]
X_66 = msci.iloc[:-1, :1]
X_6 = pd.concat([famafrench['Mkt-RF'], X_66], axis = 1)
X_6 = sm.add_constant(X_6)
model11 = sm.OLS(Y_12, X_6).fit()

# Modelo 3
Y_13 = return6_rf.iloc[:-1, :]
X_77 = msci.iloc[:-1, :1]
X_7 = pd.concat([famafrench['Mkt-RF'], X_77], axis = 1)
X_7 = sm.add_constant(X_7)
model12 = sm.OLS(Y_13, X_7).fit()

# Modelo 4
Y_14 = return7_rf.iloc[:-1, :]
X_88 =msci.iloc[:-1, :1]
X_8 = pd.concat([famafrench['Mkt-RF'], X_88], axis = 1)
X_8 = sm.add_constant(X_8)
model13 = sm.OLS(Y_14, X_8).fit()


col1, col2, col3, col4 = st.columns(4)
with col1: 
    st.write('Multifactorial model weighted for group 1:' )
    modelo10 = extract_model_summary(model10)
    st.dataframe(modelo10, use_container_width=True)

with col2: 
    st.write('Multifactorial model weighted for group 2:' )
    modelo11 = extract_model_summary(model11)
    st.dataframe(modelo11, use_container_width=True)
    
with col3: 
    st.write('Multifactorial model weighted for group 3:' )
    modelo12 = extract_model_summary(model12)
    st.dataframe(modelo12, use_container_width=True)

with col4: 
    st.write('Multifactorial modelo weighted for group 4: ')
    modelo13 = extract_model_summary(model13)
    st.dataframe(modelo13, use_container_width=True)


col1, col2 = st.columns(2)
with col1:
    st.write('Group 1:')
    st.dataframe(df_filtered, use_container_width=True)
    
    st.write('Grouop 3')
    st.dataframe(df_filtered2, use_container_width=True)
    
with col2:
    st.write('Group 2:')
    st.dataframe(df_filtered1, use_container_width=True)
    
    st.write('Grouop 4')
    st.dataframe(df_filtered3, use_container_width=True)





# python -m streamlit run "C:\Users\gadyh\OneDrive\Documentos\UNISABANA\capm_modash.py"