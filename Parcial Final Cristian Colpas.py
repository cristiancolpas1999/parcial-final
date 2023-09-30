# %% [markdown]
# Parcial Final Machine learning presentado por: Cristian Colpas

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split


# %% [markdown]
# #data fraud

# %%
fraudiden= pd.read_csv('C:/Users/cacolpas/Desktop/parcial/fraude/train_identity.csv')
fraudtran= pd.read_csv('C:/Users/cacolpas/Desktop/parcial/fraude/train_transaction.csv')

# %% [markdown]
# En esta parte se realiza la lectura de los dos dataframe de entrenamiento, train identity y train transaction

# %%
print(fraudiden.columns.values)

# %%
print(fraudtran.columns.values)

# %% [markdown]
# analisis individual de las columnas de los dos dataframe

# %%
fraudmer=pd.merge(fraudtran, fraudiden, on='TransactionID', how='left')

# %% [markdown]
# se realiza el merge para unir los dos dataframe

# %%
print(fraudmer.columns.values)

# %%
fraudmer.shape

# %% [markdown]
# #se puede ver que tenemos en el nuevo dataframe unido 434 columnas y 590540 datos

# %%
fraudmer.head()

# %% [markdown]
# utilizando .head() se puede notar que existen datos vacios, por lo cual seria interesante ver la cantidad de datos vacios que tiene en total el dataset unido

# %% [markdown]
# #

# %%
faltante=fraudmer.isnull().sum()
faltante

# %%
missing_percentage = (faltante / len(fraudmer)) * 100
missing_data_info = pd.DataFrame({
    'Columna': faltante.index,
    'Valores Faltantes': faltante.values,
    'Porcentaje Faltante': missing_percentage.values
})

# %%
missing_data_info.value_counts()


# %% [markdown]
# con isnull().sum() se puede notar que hay datos que le faltan la mayoria de datos, por lo cual seria importante darle un manejo, en este caso se dicidio eliminar estas columnas debido a que faltan la gran mayoria de ellos

# %%
#Eliminamos las columnas que tengan datos faltantes de alrededor del 70%
corte = 0.7
missing_percentage = (fraudmer.isnull().sum() / len(fraudmer)).sort_values(ascending=False)

columns_to_drop = missing_percentage[missing_percentage > corte].index

fraudmer2 = fraudmer.drop(columns=columns_to_drop)



# %% [markdown]
# eliminamos las columnas cuyos datos faltantes equivalen al 70% o mas de sus datos.

# %%
fraudmer2

# %% [markdown]
# pasamos de tener 403 columnas a 226 columnas cuando eliminamos aquellas columnas que le faltan el 70% de datos, es importante notar que en "card4" se especifica que tipo de tarjeta tuvo el usuario

# %%
fraudmer2.columns.values

# %% [markdown]
# Creamos un .values del nuevo dataset para poder saber cuales columnas quedaron despues de eliminar el 70% de columnas con datos faltantes

# %% [markdown]
# por probar, se cuentan todos los datos que tienen datos vacios, solamente para saber que informacion falta pues segun la informacion suministrada, existe cierta informacion que llegado cierto valor, seria normal que no estuviera llena o estuviera escrita con nan, por lo cual esta informacion no seria utilizada para el dataset

# %%

# Eliminar filas vacías (aquellas que contienen al menos un valor NaN)
df_clean = fraudmer2.dropna()

# Mostrar el DataFrame resultante sin filas vacías
print(df_clean)


# %% [markdown]
# como se puede notar, si se eliminan los datos faltantes, nos quedamos con menos del 10% de los datos, pues nos quedamos con solamente 40737 datos de 590540 qye habian anteriormente

# %% [markdown]
# Teniendo en cuenta que la informacion c1 al c14 indican cuantas direcciones se encuentran asociadas a la tarjeta de pago, seria interesante realizar una suma de estas variables para conocer cuantas direcciones se encuentran asociadas a cada una de estas tarjetas

# %%
fraudmer2['isFraud'] = fraudmer2['isFraud'].astype(object)

# %%
fraudmer2.describe().T

# %%
countc = fraudmer2[['C1', 'C2', 'C3','C4','C5','C6','C7','C8', 'C9','C10','C11','C12','C13','C14']].copy()

# %%
countc

# %% [markdown]
# se crea un nuevo dataframe con solo la informacion de c1 a c14

# %%
countc['count_numeric'] = (countc.iloc[:, :-1] != 0).sum(axis=1)

# %%
countc

# %%
new_df2 = countc[['count_numeric']].copy()

# %%
countc.count_numeric.value_counts()

# %% [markdown]
# en este caso obtenemos que la mayoria de datos tienes alrededor de 8 y 7 direcciones asociadas, siendo un numero menor al 50% total de los datos

# %%
fraudmer2.isFraud.value_counts()

# %%
fraudmer2.isFraud.describe()

# %% [markdown]
# segun este .describe, la mayoria de datos se encuentran en un cuartil por encima de 75%, siendo coherente debido aque solamente son 20663 datos que cometieron fraude de un total de 590540 datos

# %%
plt.title('Fraud?')
sns.countplot(x=fraudmer2.isFraud)
plt.xlabel('0=NO 1=YES')
plt.ylabel('Frequency')
plt.show()

# %% [markdown]
# se puede notar que existe un gran desbalance de datos, donde en menos del 10% de datos se detecto que se cometio fraude, siendo indicado por 1

# %%
fraudmer2.P_emaildomain.value_counts()

# %%
fraudmer2.ProductCD.value_counts()

# %%
plt.title('Tipo de compra')
sns.countplot(x=fraudmer2.ProductCD)
plt.xlabel('tipo')
plt.ylabel('Frequency')
plt.show()

# %%
fraudmer2.ProductCD.describe()

# %% [markdown]
# Revisando el tipo de producto o compra que se reviso, se puede notar que la mayoria de compras fueron de tipo W,

# %%
fraudmer2.card4.value_counts()

# %%
plt.title('tipo de tarjeta utilizada')
sns.countplot(x=fraudmer2.card4)
plt.xlabel('tipo de tarjeta')
plt.ylabel('Frequency')
plt.show()

# %%
fraudmer2.card4.describe()

# %% [markdown]
# en este caso, revisando el tipo de tarjeta utilizada, el tipo mas usado fue de tipo VISA 

# %% [markdown]
# #ahora seria realizar el analisis exploratoria del segundo dataset, el relacionado a windspeed
# 

# %%
wind= pd.read_csv('C:/Users/cacolpas/Desktop/parcial/wind speed/data_treino_dv_df_2000_2010.csv')
wind

# %%
print(wind.columns.values)

# %%
wind.shape

# %%
wind.head()

# %% [markdown]
# se cambia el formato de hora para que sea mas sencillo manejar los datos

# %%
wind['HORA (UTC)'] = wind['HORA (UTC)'].str.replace(r':\d+', '', regex=True)

# %%
wind

# %%
wind.isnull().sum()

# %%
# Eliminar filas vacías (aquellas que contienen al menos un valor NaN)
df_clean2 = wind.dropna()

# Mostrar el DataFrame resultante sin filas vacías
print(df_clean2)

# %% [markdown]
# en este caso, a diferencia del dataset anterior, no existen datos vacios, pero debido a la forma que se llaman las columnas, seria mejor renombrarlas para un mejor manejo, como tambien una mejor graficacion

# %%
nombres_nuevos = {'HORA (UTC)': 'HORA','VENTO, DIREï¿½ï¿½O HORARIA (gr) (ï¿½ (gr))': 'gr',
                  'VENTO, VELOCIDADE HORARIA (m/s)': 'm/s', 'UMIDADE REL. MAX. NA HORA ANT. (AUT) (%)': 'MAX AUT%','UMIDADE REL. MIN. NA HORA ANT. (AUT) (%)':'MIN AUT%',
                  'TEMPERATURA Mï¿½XIMA NA HORA ANT. (AUT) (ï¿½C)': 'TEMP MAX AUT','TEMPERATURA Mï¿½NIMA NA HORA ANT. (AUT) (ï¿½C)': 'TEMP MIN AUT',
                  'UMIDADE RELATIVA DO AR, HORARIA (%)':'HUMEDADRELA','PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)':'PRECION MB',
                  'PRECIPITAï¿½ï¿½O TOTAL, HORï¿½RIO (mm)': 'PRECI MM','VENTO, RAJADA MAXIMA (m/s)': 'VIENTO MAX','PRESSï¿½O ATMOSFERICA MAX.NA HORA ANT. (AUT) (mB)': 'PRESION MAX',
                  'PRESSï¿½O ATMOSFERICA MIN. NA HORA ANT. (AUT) (mB)': 'PRESION MIN'}


# %%
wind.rename(columns=nombres_nuevos, inplace=True)
wind

# %% [markdown]
# ahora que esta organizado y se redujeron los nombres, se ve mas ordenado el dataset

# %%
wind.head()

# %%
wind.HORA.value_counts()


# %%
wind.nunique()

# %%
wind.HORA.describe()

# %% [markdown]
# en el caso de contar las horas, este dataset se encuentra balanceado

# %%
wind.gr.value_counts()

# %%
from scipy.stats import kurtosis

# %%
sns.set(font_scale=1.4)
for col in wind:
    print('Column: ', col)
    print('Skew:', round(wind[col].skew(), 2))
    print('Kurtosis: ', round(wind[col].kurtosis(), 2))
    plt.figure(figsize = (14, 6))
    plt.subplot(1, 2, 1)
    wind[col].hist(grid=False)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=wind[col])
    plt.show()

# %% [markdown]
# En primer medida, viendo las graficas, estas parecen desbalanceadas, pero debido a la distribucion de los datos en los cuales algunos tienen registrados el min y el max de una variable, es entendible la tendencia de los datos

# %%
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


wind2 = wind[['MAX AUT%', 'MIN AUT%']]

wind2 = sm.add_constant(wind2)

vif = pd.DataFrame()
vif["Features"] = wind2.columns
vif["VIF"] = [variance_inflation_factor(wind2.values, i) for i in range(wind2.shape[1])]

print(vif)


# %%
wind2 = wind[['gr','MAX AUT%', 'MIN AUT%','TEMP MAX AUT','TEMP MIN AUT','HUMEDADRELA','PRECION MB','PRECI MM','VIENTO MAX','PRESION MAX','PRESION MIN']]

wind2 = sm.add_constant(wind2)

vif = pd.DataFrame()
vif["Features"] = wind2.columns
vif["VIF"] = [variance_inflation_factor(wind2.values, i) for i in range(wind2.shape[1])]

print(vif)

# %%
wind2 = wind[['PRESION MAX', 'PRESION MIN']]

wind2 = sm.add_constant(wind2)

vif = pd.DataFrame()
vif["Features"] = wind2.columns
vif["VIF"] = [variance_inflation_factor(wind2.values, i) for i in range(wind2.shape[1])]

print(vif)

# %%
from sklearn.neighbors import KNeighborsRegressor

# %%
X=wind.data
y=wind.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)


