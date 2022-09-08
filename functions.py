
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: visualizations.py : python script with data visualization functions                         -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: GPL-3.0 License                                                                            -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
# LABORATORIO 1 Paola Gomez

import pandas as pd
from os import listdir, path
from os.path import isfile, join
import numpy as np
import pandas_datareader.data as web
from scipy.optimize import minimize

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.expand_frame_repr', False)

# --------------- TRATAMIENTO DE LOS DATOS -------------------
def fun_files(path):
    '''
    Obtener la lista de los nombres de los archivos para su tratamiento
    '''
    files = [f[8:-4] for f in listdir(path) if isfile(join(path, f))]
    files = ['NAFTRAC_' + i.strftime('%Y%m%d') for i in sorted(pd.to_datetime(files))]
    return files

def fun_datafiles(files):
    '''
    Se crea un diccionario que guarda la información proveniente de los datos descrgados
    con anterioridad.
    '''

    data_files = {}
    for i in files:
        # Lee cada archivo
        data = pd.read_csv('files/' + i + '.csv', skiprows=2, header=None)
        # Listado de los nombres por columnas
        data.columns = list(data.iloc[0, :])
        # Elimina NaN's
        data = data.loc[:, pd.notnull(data.columns)]
        data = data.iloc[1:-1].reset_index(drop=True, inplace=False)
        # Eliminar , y *
        data['Precio'] = [i.replace(',', '') for i in data['Precio']]
        data['Ticker'] = [i.replace('*', '') for i in data['Ticker']]
        # Pone columnas en el orden que se necesita
        data = data.astype({'Ticker': str, 'Nombre': str, 'Peso (%)': float, 'Precio': float})
        # Peso en formato de porcentaje
        data['Peso(%)'] = data['Peso (%)'] / 100
        data_files[i] = data

    return data_files

def fun_dates(files):
    '''
    Función que regresa las fechas de los archivos  de los activos de ETF.
    '''
    dates = [j.strftime('%Y-%m-%d') for j in sorted([pd.to_datetime(i[8:]).date() for i in files])]
    return dates


def fun_tickerss(data_files, files):
    tickerss = []

    for i in files:

            tickers_a = list(data_files[i]['Ticker'])
            [tickerss.append(i + '.MX') for i in tickers_a]


    tickerss = [i.replace('GFREGIOO.MX', 'RA.MX') for i in tickerss]
    tickerss = [i.replace('MEXCHEM.MX', 'ORBIA.MX') for i in tickerss]
    tickerss = [i.replace('LIVEPOLC.1.MX', 'LIVEPOLC-1.MX') for i in tickerss]
    tickerss = [i.replace('SITESB.1', 'SITESB-1') for i in tickerss]        
    counts = pd.Series(tickerss).value_counts()
    x=counts[counts==counts.max()].index.values
    x=x.tolist()
    x.remove("MXN.MX")
    x.remove("KOFUBL.MX")
    x.sort()
    return x

def closes(tickers, start_date=None, end_date=None, freq='d'):
    '''
    Función que regresa los precios de los activos de Yahoo Finance,
    conforme a las fechas ingresadas.
    '''

    closes = web.YahooDailyReader(symbols=tickers, start=start_date, end=end_date, interval=freq).read()['Adj Close']
    closes.sort_index(inplace=True)
    return closes


# --------------- INVERSIÓN PASIVA -------------------

def inv_inicial(data_files, name, quitar, c_0, com):
    # name es el nombre del csv del cual sacaremos las ponderaciones
    
    # Creamos un dataframe con la informacion necesaria
    inicial_investing=pd.DataFrame(data_files['NAFTRAC_20200131'][['Ticker', 'Peso (%)', 'Precio']]).sort_values('Ticker')

    # Acciones compradas por ticker redondeado hacia abajo y calculo del dinero total a invertir en cada accion
    inicial_investing['# acciones']=((inicial_investing['Peso (%)']/100)*c_0/ (inicial_investing['Precio'])).apply(np.floor)
    inicial_investing['Total x ticker']=inicial_investing['Precio']*inicial_investing['# acciones']

    # Quitar tickers que no se tienen datos completos en todo el periodo
    inicial_investing=inicial_investing[inicial_investing['Ticker'].isin(quitar)==False]

    # Calcular cash
    cashh=c_0-inicial_investing['Total x ticker'].sum()+2700

    # Vector de no. de acciones de cada ticker
    noacciones=inicial_investing['# acciones']
    tickerss=inicial_investing['Ticker']
    # Linea de cash
    inicial_investing.loc['Cash']=['Cash',(cashh/c_0), '1', cashh, cashh]
    
    return inicial_investing



def portafolio_pasiva_web(dates, data_files, quitar, noacciones, web_prices, cashh):
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    portafolio_pasiva= pd.DataFrame( columns=['Fecha', 'Valor Portafolio'])
    for i in range(len(dates)):
        names=list(data_files)
        data=data_files[names[i]].sort_values('Ticker')
        data=data[data['Ticker'].isin(quitar)==False]
        data['#acciones']= noacciones.values
        data['Precio']=web_prices.loc[dates[i]].values
        data['Total x ticker']= data['Precio']*data['#acciones']
        valor=data['Total x ticker'].sum()+cashh
        portafolio_pasiva.loc[i]=[dates[i], valor]
        
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000]
    portafolio_pasiva['Valor Portafolio'].round(2)
    portafolio_pasiva['Rend']=portafolio_pasiva['Valor Portafolio'].pct_change()
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000, 0]
    portafolio_pasiva['Rend acumulado']= ''
    for i in range(len(dates)-1):
        portafolio_pasiva['Rend acumulado'][(i+1)]=portafolio_pasiva['Rend'][i]+portafolio_pasiva['Rend'][(i+1)]
    
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000, 0, 0]
    
    return portafolio_pasiva



def portafolio_pasiva_csv(dates, data_files, quitar, noacciones, cashh):
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    portafolio_pasiva= pd.DataFrame( columns=['Fecha', 'Valor Portafolio'])
    for i in range(len(dates)):
        names=list(data_files)
        data=data_files[names[i]].sort_values('Ticker')
        data=data[data['Ticker'].isin(quitar)==False]
        data['#acciones']= noacciones.values
        #data['Precio']=web_prices.loc[dates[i]].values
        data['Total x ticker']= data['Precio']*data['#acciones']
        valor=data['Total x ticker'].sum()+cashh
        portafolio_pasiva.loc[i]=[dates[i], valor]
        
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000]
    portafolio_pasiva['Valor Portafolio'].round(2)
    portafolio_pasiva['Rend']=portafolio_pasiva['Valor Portafolio'].pct_change()
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000, 0]
    portafolio_pasiva['Rend acumulado']= ''
    for i in range(len(dates)-1):
        portafolio_pasiva['Rend acumulado'][(i+1)]=portafolio_pasiva['Rend'][i]+portafolio_pasiva['Rend'][(i+1)]
    
    portafolio_pasiva.loc[0]=['2020-01-31', 1000000, 0, 0]
    
    return portafolio_pasiva


# --------------- INVERSIÓN ACTIVA -------------------

def portemvrs(web_prices, rf, tickers, dates, c_0,web_prices29):
    
    ret= np.log(web_prices)-np.log(web_prices.shift(1))
    ret=ret.dropna()
    summary=pd.DataFrame(data={'Media':252*ret.mean(),'Vol':(252**.5)*ret.std()})
    
    # Funcion a minimizar
    def menos_rs(w, Eind, rf, Sigma):
        ep = Eind.dot(w)
        sp = (w.transpose().dot(Sigma).dot(w))**0.5
        rs = (ep - rf) / sp
        return -rs
    
    ## Construcción de parámetros
    # 1. Sigma: matriz de varianza-covarianza Sigma = S.dot(corr).dot(S)

    corr=ret.corr()
    S=np.diag(summary.loc[:,'Vol'])
    Sigma= S.dot(corr).dot(S).astype(float)

    # 2. Eind: rendimientos esperados activos individuales
    Eind=  summary.loc[:,'Media'].values.astype(float)
    
    # Número de activos
    n= len(Eind)

    # Dato inicial
    w0= np.ones(n)/n

    # Cotas de las variables
    bnds=((0,1),)*n

    # Restricciones
    cons= {"type": "eq", "fun": lambda w:w.sum()-1}
    
    # Portafolio EMV
    emv=minimize(fun=menos_rs,
                    x0=w0,
                    args=(Eind, rf, Sigma),
                    bounds=bnds,
                    constraints=cons,
                    tol=1e-10)
    
    # Pesos, rendimiento, riesgo y razon de sharpe del portafolio de mínima varianza
    w_emv=emv.x
    e_emv=Eind.dot(w_emv)
    s_emv=(w_emv.T.dot(Sigma).dot(w_emv))**0.5
    rs_emv= (e_emv-rf)/s_emv
    # e_emv, s_emv, rs_emv
    pon=pd.DataFrame(data={'Activo':tickers,'Ponderacion': w_emv})
    pon.round(5).sort_values('Ponderacion')
    a=pon[pon['Ponderacion']>0]
    a['Precio']=web_prices29.loc['2021-03-01',a['Activo']].values
    a['# acciones']= (a['Ponderacion']*c_0/a['Precio']).apply(np.floor)
    a['Total x ticker']=a['Precio']*a['# acciones']
    a=a[a['# acciones']>0]
    return a


def portafolio_activo(portafolio0, web_prices1, c_0, com, cash0):
     # Cash
    cashhh=c_0-portafolio0['Total x ticker'].sum()
    
    # Fechas
    dates1=web_prices1.index

    # Portafolio incial
    portafolio=portafolio0

    # empty data frame Transacciones
    transacciones=pd.DataFrame(columns=['Fecha', 'Tickers', 'Precio', '# Titulos', 'Comision', '$ Total', 'Tipo'])

    # empty dataframe Portafolios
    portafolios=pd.DataFrame(columns=['Fecha', 'Total', 'Cash'])
    
        # Ciclo fechas
    for i in range(len(dates1)-1):

        # Calculo de los cambios en precios
        difs=web_prices1.loc[dates1[i+1], portafolio['Activo']]/web_prices1.loc[dates1[i], portafolio['Activo']]-1

        # Ventas de activos que perdieron mas del 5% de su valor
        ventas=difs[difs<-0.05]
        vender=portafolio.loc[portafolio['Activo'].isin(ventas.index.values)]
        vender['dates']=dates1[i+1]

        # Data Frame con las transacciones del periodo de ventas
        df_ventas=pd.DataFrame({'Fecha':vender['dates'],
                  'Tickers':vender['Activo'],
                  'Precio':vender['Precio'],
                  '# Titulos':np.floor(vender['# acciones']*0.025),
                  'Comision': np.floor(vender['# acciones']*0.025)*vender['Precio']*com,
                  '$ Total': (np.floor(vender['# acciones']*0.025)*vender['Precio'])*(1-com),
                  'Tipo': 'Venta'})

        # Parte compras activos que ganaron mas del 5% de su valor
        compras=difs[difs>0.05]
        comprar=portafolio.loc[portafolio['Activo'].isin(compras.index.values)]
        comprar['dates']=dates1[i+1]
        comprar['crecimiento']=compras.values

        # Actualizacion del data frame transacciones y del cashhh y calculo de titulos de ventas
        transacciones=pd.concat([transacciones, df_ventas])
        cashhh= cashhh+df_ventas['$ Total'].sum()
        titulos_venta=portafolio.loc[portafolio['Activo'].isin(ventas.index.values)]['# acciones']



        # Data Frame con las transacciones del periodo
        df_compras=pd.DataFrame({'Fecha':comprar['dates'],
                    'Tickers':comprar['Activo'],
                    'Precio':comprar['Precio'],
                    '# Titulos':np.floor(comprar['# acciones']*0.025),
                    'Comision': np.floor(comprar['# acciones']*0.025)*comprar['Precio']*com,
                    '$ Total': (np.floor(comprar['# acciones']*0.025)*comprar['Precio'])*(1+com),
                     'Tipo': 'Compra',
                    'Crecimiento': comprar['crecimiento']})
        # Cambiamos el orden de las compras para dar prioridad a activos con mayor crecimiento
        df_compras=df_compras.sort_values('Crecimiento', ascending=False)

        # Compramos los activos que nos alcance el cashhh
        tickers_compra=[]
        for i in range(len(df_compras)):
            dinero=df_compras.iloc[i,5]

            if dinero<cashhh:
                    transacciones=pd.concat([transacciones, df_compras.iloc[:,:-1]])
                    cashhh=cashhh-dinero
                    tickers_compra.append(df_compras.iloc[i,1])

            else:
                    pass

        titulos_compra=portafolio.loc[portafolio['Activo'].isin(tickers_compra)]['# acciones'].values



        # Nuevo portafolio actualizado ventas, compras y precio
        portafolio['Precio']=web_prices1.loc[dates1[i+1], portafolio['Activo']].values
        portafolio.loc[portafolio['Activo'].isin(ventas.index.values), '# acciones']=titulos_venta*0.975
        portafolio.loc[portafolio['Activo'].isin(tickers_compra), '# acciones']=titulos_compra*1.025
        portafolio['Total x ticker']=portafolio['Precio']*portafolio['# acciones']

        # Valor de los portafolios diarios
        portafolios.loc[i]=[dates1[i+1], portafolio['Total x ticker'].sum(), cashhh]

    #portafolios =portafolios.iloc[1:]
    portafolios['Rend']=portafolios['Total'].pct_change()
    portafolios.loc[0]=['2021-03-02', 1000000,cash0, 0]
    portafolios['Rend acumulado']= ''
    portafolios=portafolios.reset_index(drop=True)
    
    for i in range(len(portafolios)-1):
        portafolios['Rend acumulado'][(i+1)]=portafolios['Rend'][i]+portafolios['Rend'][(i+1)]
    portafolios.loc[0]=['2021-03-02', 1000000,cash0, 0, 0]
    
    return(portafolios)



def transacciones_activo(portafolio0, web_prices1, c_0, com):
     # Cash
    cashhh=c_0-portafolio0['Total x ticker'].sum()
    
    # Fechas
    dates1=web_prices1.index

    # Portafolio incial
    portafolio=portafolio0

    # empty data frame Transacciones
    transacciones=pd.DataFrame(columns=['Fecha', 'Tickers', 'Precio', '# Titulos', 'Comision', '$ Total', 'Tipo'])

    # empty dataframe Portafolios
    portafolios=pd.DataFrame(columns=['Fecha', 'Total', 'Cash'])
    
        # Ciclo fechas
    for i in range(len(dates1)-1):

        # Calculo de los cambios en precios
        difs=web_prices1.loc[dates1[i+1], portafolio['Activo']]/web_prices1.loc[dates1[i], portafolio['Activo']]-1

        # Ventas de activos que perdieron mas del 5% de su valor
        ventas=difs[difs<-0.05]
        vender=portafolio.loc[portafolio['Activo'].isin(ventas.index.values)]
        vender['dates']=dates1[i+1]

        # Data Frame con las transacciones del periodo de ventas
        df_ventas=pd.DataFrame({'Fecha':vender['dates'],
                  'Tickers':vender['Activo'],
                  'Precio':vender['Precio'],
                  '# Titulos':np.floor(vender['# acciones']*0.025),
                  'Comision': np.floor(vender['# acciones']*0.025)*vender['Precio']*com,
                  '$ Total': (np.floor(vender['# acciones']*0.025)*vender['Precio'])*(1-com),
                  'Tipo': 'Venta'})

        # Parte compras activos que ganaron mas del 5% de su valor
        compras=difs[difs>0.05]
        comprar=portafolio.loc[portafolio['Activo'].isin(compras.index.values)]
        comprar['dates']=dates1[i+1]
        comprar['crecimiento']=compras.values

        # Actualizacion del data frame transacciones y del cashhh y calculo de titulos de ventas
        transacciones=pd.concat([transacciones, df_ventas])
        cashhh= cashhh+df_ventas['$ Total'].sum()
        titulos_venta=portafolio.loc[portafolio['Activo'].isin(ventas.index.values)]['# acciones']



        # Data Frame con las transacciones del periodo
        df_compras=pd.DataFrame({'Fecha':comprar['dates'],
                    'Tickers':comprar['Activo'],
                    'Precio':comprar['Precio'],
                    '# Titulos':np.floor(comprar['# acciones']*0.025),
                    'Comision': np.floor(comprar['# acciones']*0.025)*comprar['Precio']*com,
                    '$ Total': (np.floor(comprar['# acciones']*0.025)*comprar['Precio'])*(1+com),
                     'Tipo': 'Compra',
                    'Crecimiento': comprar['crecimiento']})
        # Cambiamos el orden de las compras para dar prioridad a activos con mayor crecimiento
        df_compras=df_compras.sort_values('Crecimiento', ascending=False)

        # Compramos los activos que nos alcance el cashhh
        tickers_compra=[]
        for i in range(len(df_compras)):
            dinero=df_compras.iloc[i,5]

            if dinero<cashhh:
                    transacciones=pd.concat([transacciones, df_compras.iloc[:,:-1]])
                    cashhh=cashhh-dinero
                    tickers_compra.append(df_compras.iloc[i,1])

            else:
                    pass

        titulos_compra=portafolio.loc[portafolio['Activo'].isin(tickers_compra)]['# acciones'].values



        # Nuevo portafolio actualizado ventas, compras y precio
        portafolio['Precio']=web_prices1.loc[dates1[i+1], portafolio['Activo']].values
        portafolio.loc[portafolio['Activo'].isin(ventas.index.values), '# acciones']=titulos_venta*0.975
        portafolio.loc[portafolio['Activo'].isin(tickers_compra), '# acciones']=titulos_compra*1.025
        portafolio['Total x ticker']=portafolio['Precio']*portafolio['# acciones']

        # Valor de los portafolios diarios
        portafolios.loc[i]=[dates1[i+1], portafolio['Total x ticker'].sum(), cashhh]

    portafolios =portafolios.iloc[1:]
    return(transacciones)


def resultados(porti, web_prices1, portafolio0, rf, y, web_prices, inicial_investing):
    
    # Funcion radio de sharpe
    def rsharpe(web_prices, w, rf):
        ret= np.log(web_prices)-np.log(web_prices.shift(1))
        ret=ret.dropna()
        summary=pd.DataFrame(data={'Media':252*ret.mean(),'Vol':(252**.5)*ret.std()})
        Eind=  summary.loc[:,'Media'].values.astype(float)

        corr=ret.corr()
        S=np.diag(summary.loc[:,'Vol'])
        Sigma= S.dot(corr).dot(S).astype(float)

        ep = Eind.dot(w)
        sp = (w.transpose().dot(Sigma).dot(w))**0.5
        rs = (ep - rf) / sp
        return rs
    
    # w's
    wpas=inicial_investing.iloc[:-1,1].values/100
    wact=portafolio0['Ponderacion'].values
    
    # Dataframe
    dfa=pd.DataFrame({'Medida':['rend_m', 'rend_c', 'sharpe'],
              'Descripcion':['Rendimiento Promedio Mensual', 'Rendimiento mensual acumulado', 'Sharpe Ratio'],
              'inv_activa': [porti['Rend'].mean(), porti['Rend acumulado'].sum(), -rsharpe(web_prices1[portafolio0['Activo'].values], wact, rf)],
              'inv_pasiva': [y['Rend'].mean(), y['Rend acumulado'].sum(), rsharpe(web_prices, wpas, rf)]})
    return dfa
