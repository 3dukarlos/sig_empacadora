from tkinter import Y
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from PIL import Image
import copy
import plost
import statsmodels.api as sm
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from streamlit_option_menu import option_menu
from datetime import datetime

cmap = copy.copy(plt.cm.get_cmap("Blues"))
cmap.set_under("white")


# st.set_page_config(  # Alternate names: setup_page, page, layout
# 	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
# 	initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
# 	page_title=None,  # String or None. Strings get appended with "• Streamlit". 
# 	page_icon=None,  # String, anything supported by st.image, or None.
# )

st.set_page_config(
    page_title="(sig) Agro",
    layout="wide",
    initial_sidebar_state="expanded",
)

colimg1, colimg2 = st.columns((1,1.8))
image = Image.open('huamani.png')
colimg1.image(image, width=327)

#st.set_page_config(page_title='Survey Results', layout='wide')
colimg2.header('**(sig) Sistema Integrado de Gestión**')
colimg2.subheader('KPI Agro')
colimg2.markdown("""
*v. 1.0*
""")

#---------------------------------#
# Leer archivo
db_GENERAL = 'data_GENERAL.xlsx'
db_PROCESO = 'data_PROCESO.xlsx'
db_PROYECCION = 'data_PROYECCION.xlsx'

@st.cache(allow_output_mutation=True)
def load_files(filename, sheet):
    df = pd.read_excel(filename, sheet_name=sheet, engine='openpyxl')
    return df


df = load_files(db_GENERAL, 'db_costos')
df_COSECHA = load_files(db_GENERAL, 'db_cosecha')
df_PROCESO = load_files(db_PROCESO, 'db_proceso')
df_DESPACHOS = load_files(db_PROCESO, 'db_despachos')
df_PROYECCION = load_files(db_PROYECCION, 'db_proyeccion')


# horizontal menu
selected = option_menu(None, ["Summary", "Details", "Plots", 'Harvest', 'Schedule', 'Packing', 'COMEX', 'Profits' ], 
    icons=['house', 'kanban', 'bar-chart-fill', 'table', 'calendar3-range', 'box-seam', 'calendar-check-fill', 'currency-dollar'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
        styles={
        #"container": {"padding": "0!important", "background-color": "#fafafa"},
        #"icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "12px"},
        #, "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#0D33CE"},
    }
)

#---------------------------------#
# Funcion de filtrado del dataframa - Menu multiselect
def df_filtered(
    df: pd.DataFrame,  # Source dataframe
    #f_date_range: [int, int],  # Current value of an ST date slider
    f_fundo: list = [],  # Current value of an ST multi-select
    f_VARIEDAD: list = [],  # Current value of another ST multi-select
    f_lote: list = [],  # Current value of another ST multi-select
    f_tipogasto: list = []  # Current value of another ST multi-select
) -> pd.DataFrame:
    dff = df.loc[(df['UNIDAD NEGOCIO'].isin(f_fundo))].reset_index(drop=True)
    if len(f_VARIEDAD) > 0:
        dff = dff.loc[(dff['VARIEDAD'].isin(f_VARIEDAD))].reset_index(drop=True)
    if len(f_lote) > 0:
        dff = dff.loc[(dff['CONSUMIDOR'].isin(f_lote))].reset_index(drop=True)
    if len(f_tipogasto) > 0:
        dff = dff.loc[(dff['TIPO GASTO'].isin(f_tipogasto))].reset_index(drop=True)
    return dff

def df2_filtered(
    df: pd.DataFrame,  # Source dataframe
    f_tipogasto: list = [],  # Current value of an ST multi-select
    f_actilabor: list = []  # Current value of an ST multi-select
) -> pd.DataFrame:
    dff = df.loc[(df['TIPO GASTO'].isin(f_tipogasto))].reset_index(drop=True)
    if len(f_actilabor) > 0:
        dff = dff.loc[(dff['ACTIVIDAD / LABOR'].isin(f_actilabor))].reset_index(drop=True)
    return dff

# Funcion estilo (numeros negativos)
def highlight_max(cell):
    if type(cell) != str and cell < 0:
        return 'color: red'
    else:
        return 'color: black'

def color_nan_white(val):
    """Color the nan text white"""
    if np.isnan(val):
        return 'color: #C6EFCE'

def color_nan_white_background(val):
    """Color the nan cell background white"""
    if np.isnan(val):
        return 'background-color: #C6EFCE'


fundo = df['UNIDAD NEGOCIO'].unique().tolist()
VARIEDAD = df['VARIEDAD'].unique().tolist()
CONSUMIDOR = df['CONSUMIDOR'].unique().tolist()
labor = df['TIPO GASTO'].unique().tolist()

dff = df_filtered(df, f_fundo=fundo, f_VARIEDAD=VARIEDAD, f_lote=CONSUMIDOR, f_tipogasto=labor)


#---------------------------------#
# Menu multiselect - opciones de filtrado
col1 = st.sidebar
col1.header('Opciones de filtrado')


with col1.expander("Unidad de Negocio"):
    fundo1 = st.multiselect(
        'Seleccionar la Unidad de Negocio:',
        fundo,
        default = None
        )

dff = df_filtered(dff, f_fundo=fundo1, f_VARIEDAD=VARIEDAD, f_lote=CONSUMIDOR, f_tipogasto=labor)
VARIEDAD1 = dff['VARIEDAD'].unique().tolist()

with col1.expander("Variedad"):
    VARIEDAD2 = st.multiselect(
        'Seleccionar la Variedad:',
        VARIEDAD1,
        default = None
        )          


dff = df_filtered(dff, f_fundo=fundo1, f_VARIEDAD=VARIEDAD2, f_lote=CONSUMIDOR, f_tipogasto=labor)
CONSUMIDOR1 = dff['CONSUMIDOR'].unique().tolist()

with col1.expander("Consumidor"):
    lote = st.multiselect(
        'Seleccionar el Consumidor/Lote:',
        CONSUMIDOR1,
        default = CONSUMIDOR1
        )


dff = df_filtered(dff, f_fundo=fundo1, f_VARIEDAD=VARIEDAD2, f_lote=CONSUMIDOR1, f_tipogasto=labor)
labor1 = dff['TIPO GASTO'].unique().tolist()

with col1.expander("Tipo de Gasto"):
    actividades1 = st.multiselect(
        'Seleccionar el Tipo de Gasto:',
        labor1,
        default = labor1
        )  


df_acumulado = df.groupby(['UNIDAD NEGOCIO', 'VARIEDAD', 'CONSUMIDOR', 'TIPO GASTO'])['IMPORTE'].sum().reset_index()
df_semanal = df.groupby(['UNIDAD NEGOCIO', 'VARIEDAD', 'CONSUMIDOR', 'TIPO GASTO', 'AÑO', 'SEMANA'])['IMPORTE'].sum().reset_index()
df_actilabor = df.groupby(['UNIDAD NEGOCIO', 'VARIEDAD', 'CONSUMIDOR', 'TIPO GASTO', 'AÑO', 'SEMANA', 'ACTIVIDAD / LABOR'])['IMPORTE'].sum().reset_index()
df_aqfermat = df.groupby(['UNIDAD NEGOCIO', 'VARIEDAD', 'CONSUMIDOR', 'TIPO GASTO', 'SUBTIPO1', 'SUBTIPO2', 'AÑO', 'SEMANA', 'ACTIVIDAD / LABOR'])['IMPORTE'].sum().reset_index()

#---------------------------------#
# Dataframe filtrado
df_HEADER = df_acumulado[ (df_acumulado['UNIDAD NEGOCIO'].isin(fundo1)) & (df_acumulado['VARIEDAD'].isin(VARIEDAD2)) 
 & (df_acumulado['CONSUMIDOR'].isin(lote)) & (df_acumulado['TIPO GASTO'].isin(actividades1)) ]

df_PLOT2 = df_HEADER.groupby(['VARIEDAD', 'TIPO GASTO'])['IMPORTE'].sum().reset_index()

### --------------------------------
df_FOOTER = df_semanal[ (df_semanal['UNIDAD NEGOCIO'].isin(fundo1)) & (df_semanal['VARIEDAD'].isin(VARIEDAD2)) 
 & (df_semanal['CONSUMIDOR'].isin(lote)) & (df_semanal['TIPO GASTO'].isin(actividades1)) ]

df_TIPOGASTO = df_actilabor[ (df_actilabor['UNIDAD NEGOCIO'].isin(fundo1)) & (df_actilabor['VARIEDAD'].isin(VARIEDAD2)) 
 & (df_actilabor['CONSUMIDOR'].isin(lote)) & (df_actilabor['TIPO GASTO'].isin(actividades1)) ]

df_TIPOGASTO_OTROS = df_aqfermat[ (df_aqfermat['UNIDAD NEGOCIO'].isin(fundo1)) & (df_aqfermat['VARIEDAD'].isin(VARIEDAD2)) 
 & (df_aqfermat['CONSUMIDOR'].isin(lote)) & (df_aqfermat['TIPO GASTO'].isin(actividades1)) ]


#---------------------------------#
# MANO DE OBRA / pivot - filter

df_MO = df_TIPOGASTO[(df_TIPOGASTO['TIPO GASTO']=='(A) MANO DE OBRA')]
df_AQ = df_TIPOGASTO_OTROS[(df_TIPOGASTO_OTROS['TIPO GASTO']=='(D) AGROQUIMICOS')]
df_FE = df_TIPOGASTO_OTROS[(df_TIPOGASTO_OTROS['TIPO GASTO']=='(C) FERTILIZANTES')]
tipogasto = df_MO['TIPO GASTO'].unique().tolist()
act_labor = df_MO['ACTIVIDAD / LABOR'].unique().tolist()
act_subtipo1 = df_TIPOGASTO_OTROS['SUBTIPO1'].unique().tolist()



#---------------------------------#
    
if selected == "Summary":

    # Expander
    expander_bar1 = st.expander("Costos y Gastos - por Tipo (acumulado)")

    # Render
    if df_HEADER.shape[0]:
        pivot_SUMM = pd.pivot_table(df_HEADER, index=['VARIEDAD', 'TIPO GASTO'], columns='CONSUMIDOR', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
        expander_bar1.dataframe(pivot_SUMM.style\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
        )


    st.markdown('---')


    ### --- COLUMNS ---
    col2, col3 = st.columns((1.3,1))

    with col2:
        st.write("**Costos Acumulados por CONSUMIDOR**")

        fig = px.bar(df_HEADER, y='CONSUMIDOR', x='IMPORTE', template="simple_white",
        color ='TIPO GASTO', barmode='stack', height=350, width=600)
        fig.update_traces(textfont_size=10, textangle=90, textposition="inside")
        fig.update_layout(
        legend=dict(
            title=None, orientation="v", y=1, x=1, 
            font=dict(
                family="Segoe UI Symbol",
                size=11.5,
                color="black"
                ),
            borderwidth=0
            )
        )
        st.plotly_chart(fig)

    with col3:
        #st.write("Hello **world**!")
        st.write("**Tipos de Gasto por CONSUMIDOR**")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        plost.donut_chart(
            data=df_PLOT2,
            color='TIPO GASTO',
            theta='IMPORTE'
            )
    # pie_chart = px.pie(df_grouped1,
    #         values='Tipo_Gasto',
    #         names='TIPO GASTO')
    # st.plotly_chart(pie_chart)


elif selected == "Details":

    #tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["DETALLE GENERAL POR TIPO DE GASTO (semanal)", "(A) MANO DE OBRA",\
        "(D) AGROQUIMICOS","(C) FERTILIZANTES", "(F) ENVASES Y EMBAJALES"])


    #pivot general de Tipos de Gasto
    tab1.caption('Pivot General de Tipos de Gasto - **Semana de miércoles a martes (según planilla)**')

    if df_FOOTER.shape[0]:
        #tab1
        pivot_WEEK = pd.pivot_table(df_FOOTER, index=['CONSUMIDOR', 'TIPO GASTO'], columns='SEMANA', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')

        tab1.dataframe(pivot_WEEK.style\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
        )
        
    #tab2
        act_laborCOMBO = tab2.multiselect('Actividad Labor:', act_labor, act_labor)

        # act_laborCOMBO = tab2.selectbox('Actividad Labor:', act_labor)
        # fx filtrado dt LABORES DE MANO DE OBRA
        dff2_ = df2_filtered(df_MO, f_tipogasto=tipogasto, f_actilabor=act_labor)

        #df filtered
        df_LABORES = df_MO[ (df_MO['ACTIVIDAD / LABOR'].isin(act_laborCOMBO)) ]

        #grafico de barras verticales por labores de MANO DE OBRA
        tab2.write("**Actividad / Labor por CONSUMIDOR**")
        fig = px.bar(df_LABORES.groupby(['CONSUMIDOR', 'ACTIVIDAD / LABOR'])['IMPORTE'].sum().reset_index(), y='IMPORTE', x='CONSUMIDOR', template="simple_white",
        color ='ACTIVIDAD / LABOR', barmode='stack', height=500, width=1000)
        fig.update_traces(textfont_size=15, textangle=0,
        textposition="inside", texttemplate = "%{value:,s}",)
        fig.update_layout(
            legend=dict(
                title=None,
                yanchor="bottom",
                y=0.1,
                xanchor="left",
                x=1,
                font=dict(
                    family="Segoe UI Symbol",
                    size=11.5,
                    color="black"
                    ),
                    borderwidth=0
            )
        )
        fig.update_layout(autosize=False)
        tab2.plotly_chart(fig)

        tab2.markdown('---')

        #pivot de labores de MANO DE OBRA

        tab2.caption('Pivot de labores de MANO DE OBRA por semana')

        if df_LABORES.shape[0]:
            pivot_TIPOGASTOMO = pd.pivot_table(df_LABORES, index=['VARIEDAD', 'CONSUMIDOR', 'ACTIVIDAD / LABOR'], columns='SEMANA', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
            tab2.dataframe(pivot_TIPOGASTOMO.style\
                .background_gradient(axis='index', low=0, high=1.0)\
                .applymap(highlight_max).format('{:,.2f}')\
                .applymap(lambda x: color_nan_white(x))\
                .applymap(lambda x: color_nan_white_background(x))
                )
        
    #tab3
        tab3.caption('Pivot de Tipo de AGROQUIMICOS por semana')
        if df_AQ.shape[0]:
            pivot_TIPOGASTOAQ = pd.pivot_table(df_AQ, index=['VARIEDAD', 'CONSUMIDOR', 'SUBTIPO1'], columns='SEMANA', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
            tab3.dataframe(pivot_TIPOGASTOAQ.style\
                .background_gradient(axis='index', low=0, high=1.0)\
                .applymap(highlight_max).format('{:,.2f}')\
                .applymap(lambda x: color_nan_white(x))\
                .applymap(lambda x: color_nan_white_background(x))
                )
        tab3.markdown('---')
        act_aqfermat = tab3.multiselect('Tipo de Agroquimico:', act_subtipo1, act_subtipo1)
        df_AQdetails = df_AQ[ (df_AQ['SUBTIPO1'].isin(act_aqfermat)) ]

        tab3.caption('Pivot de Productos por tipo de AGROQUIMICOS por semana')
        if df_AQdetails.shape[0]:
            pivot_TIPOGASTOAQdetails = pd.pivot_table(df_AQdetails, index=['VARIEDAD', 'CONSUMIDOR', 'SUBTIPO2'], columns='SEMANA', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
            tab3.dataframe(pivot_TIPOGASTOAQdetails.style\
                .background_gradient(axis='index', low=0, high=1.0)\
                .applymap(highlight_max).format('{:,.2f}')\
                .applymap(lambda x: color_nan_white(x))\
                .applymap(lambda x: color_nan_white_background(x))
                )

        tab3.markdown('---')
        expander_AQ = st.expander("Costos Agroquimicos - por Producto (acumulado)")
        if df_AQdetails.shape[0]:
            pivot_TIPOGASTOAQdetails = pd.pivot_table(df_AQdetails, index=['VARIEDAD', 'SUBTIPO2'], columns='CONSUMIDOR', values='IMPORTE', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
            tab3.dataframe(pivot_TIPOGASTOAQdetails.style\
                .background_gradient(axis='index', low=0, high=1.0)\
                .applymap(highlight_max).format('{:,.2f}')\
                .applymap(lambda x: color_nan_white(x))\
                .applymap(lambda x: color_nan_white_background(x))
                )

elif selected == "Plots":

    #max semana de los costos
    # st.write(df_MO['SEMANA'].max())
    year = 2022
    df_COSECHAFT = df_COSECHA[ (df_COSECHA['AÑO'].isin([year])) & (df_COSECHA['LOTE'].isin(df_MO['CONSUMIDOR'])) & (df_COSECHA['VARIEDAD'].isin(df_MO['VARIEDAD'])) & (df_COSECHA['SEMANA'] <= df_MO['SEMANA'].max()) ]
    #df cosecha real x semana
    df_COSEgrouped = df_COSECHAFT.groupby(['LOTE','VARIEDAD','SEMANA'])['peso_neto'].sum().reset_index()

    if df_COSECHAFT.shape[0]:
        st.caption('Pivot de COSECHA por semana (NISIRA)')
        pivot_COSECHA = pd.pivot_table(df_COSECHAFT.sort_values(by="VARIEDAD"), index=['VARIEDAD','LOTE'], columns='SEMANA', values='peso_neto', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
        st.dataframe(pivot_COSECHA.style\
            .background_gradient(axis='index', low=0, high=1.0)\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
            )

    st.markdown('---')

    #grafico de lineas de Labor por semana
    st.caption('Dispersión de labores de MANO DE OBRA por consumidor')

    if df_MO.shape[0]:
        fig = px.scatter(df_MO.sort_values(by="CONSUMIDOR"), x='SEMANA', y='IMPORTE', template="simple_white",
        color ='ACTIVIDAD / LABOR', height=500, width=1000, facet_col='CONSUMIDOR', facet_col_wrap=4)

        #model prediction
        df_pred = df_MO.sort_values(by="SEMANA")
        model = sm.OLS(df_pred["IMPORTE"], sm.add_constant(df_pred["SEMANA"])).fit()

        #create the trace to be added to all facets
        trace = go.Scatter(x=df_pred["SEMANA"], y=model.predict(),
                        line_color="black", name="OLS stat")

        # give it a legend group and hide it from the legend
        trace.update(legendgroup="trendline", showlegend=False)
        fig.add_trace(trace, row="all", col="all", exclude_empty_subplots=True)
        # set only the last trace added to appear in the legend
        # `selector=-1` introduced in plotly v4.13
        fig.update_traces(selector=-1, showlegend=True)

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.update_traces(textfont_size=10)
        fig.update_layout(
            legend=dict(
                title=None,
                yanchor="bottom",
                y=0.1,
                xanchor="left",
                x=1,
                font=dict(
                    family="Segoe UI Symbol",
                    size=11.5,
                    color="black"
                    ),
                    borderwidth=0
            )
        )
        fig.update_layout(autosize=False)

        #add trace cosecha real
        # fig2 = px.scatter(
        #     df_COSEgrouped.sort_values('SEMANA'),
        #     x='SEMANA',
        #     y='peso_neto',
        #     # x=['08.01','08.02','08.03','08.04','08.05','08.06'],
        #     # y=[80,100,110,90,120,130],
        #     color_discrete_sequence=['red'],
        #     labels=dict(SEMANA="SEMANA", peso_neto="PESO (tn)", VARIEDAD="VARIEDAD"))
        #     # x=df_PROYgrouped['FECHAPROG'],
        #     # y=df_PROYgrouped['PESO']
        #     # row='all', col='all', exclude_empty_subplots=True)
        # fig2.update_traces(mode="lines+markers", name='TN reales')#, col='all', exclude_empty_subplots=True)
        # fig2.update_traces(line_color='red', line_width=2)
        # fig2.update_traces(selector=-1, showlegend=True)
        # ####################
        # fig.add_traces(fig2._data)

        st.plotly_chart(fig)
        #tab2.write(model.summary())

    if df_MO.shape[0]:
        fig = px.scatter(df_COSEgrouped.sort_values(by=['LOTE','SEMANA']), x='SEMANA', y='peso_neto', template="simple_white",
        color_discrete_sequence=['#3d41c6'], labels=dict(SEMANA="SEMANA", peso_neto="PESO (tn)", VARIEDAD="VARIEDAD"),
        height=500, width=900, facet_col='LOTE', facet_col_wrap=4)

        df_COSEgrouped.sort_values('LOTE')
        fig.update_traces(textfont_size=10)
        fig.update_layout(
            legend=dict(
                title=None,
                yanchor="top",
                y=1,
                xanchor="left",
                x=1,
                font=dict(
                    family="Segoe UI Symbol",
                    size=11.5,
                    color="black"
                    ),
                    borderwidth=0
            )
        )
        fig.update_layout(autosize=False)
        fig.update_traces(mode="lines+markers", name='TN reales')
        fig.update_traces(line_color='#3d41c6', line_width=2)
        fig.update_traces(selector=-1, showlegend=True)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        ####################

        st.plotly_chart(fig)
        #tab2.write(model.summary())    


elif selected == "Harvest":

    if df_MO.shape[0]:
        #expander COSECHA gral
        expander_bar1 = st.expander("Resumen Cosecha KG - [Todos los Cultivos]")
        #tab3 Reporte general de Cosecha kg
        pivot_COSECHA_GRAL = pd.pivot_table(df_COSECHA.sort_values(by="VARIEDAD"), index=['VARIEDAD','LOTE'], columns='SEMANA_CALENDAR', values='peso_neto', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
        expander_bar1.dataframe(pivot_COSECHA_GRAL.style\
            .background_gradient(axis='index', low=0, high=1.0)\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
            )

        #usando los filtros del sidebar    
        #filter year
        year2122 = list(reversed(range(2021,2023)))
        year_selection = st.selectbox('Año:', year2122)

        df_COSECHA = df_COSECHA [ (df_COSECHA['AÑO'].isin([year_selection])) & (df_COSECHA['UNIDAD NEGOCIO'].isin(fundo1)) & (df_COSECHA['VARIEDAD'].isin(VARIEDAD2)) ]
        
        number_of_result = df_COSECHA.shape[0]
        st.markdown(f'*Available Result: {number_of_result}*')

        st.caption('Pivot General de kg Cosechados por semana (CALENDARIO)')
        pivot_COSECHA_GRAL = pd.pivot_table(df_COSECHA.sort_values(by="VARIEDAD"), index=['VARIEDAD','LOTE'], columns='SEMANA_CALENDAR', values='peso_neto', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
        st.dataframe(pivot_COSECHA_GRAL.style\
            .background_gradient(axis='index', low=0, high=1.0)\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
            )

elif selected == "Packing":

    #filter year
    year2122 = list(reversed(range(2021,2023)))
    year_selection = st.selectbox('Año:', year2122)

    #mask = (df_PROCESO['SEMANA'].between(*semana_selection)) & (df_PROCESO['UNIDAD NEGOCIO'].isin(office_selection)) & (df_PROCESO['VARIEDAD'].isin(variety_selection))
    #df_PROCESO = df_PROCESO[ (df_COSECHA['LOTE'].isin(df_MO['CONSUMIDOR'])) & (df_COSECHA['VARIEDAD'].isin(df_MO['VARIEDAD'])) ]
    df_PROCESO2122 = df_PROCESO [ (df_PROCESO['AÑO'].isin(year2122)) & (df_PROCESO['UNIDAD NEGOCIO'].isin(fundo1)) & (df_PROCESO['VARIEDAD'].isin(VARIEDAD2)) & (df_PROCESO['LOTE'].isin(lote)) ]
    df_PROCESO = df_PROCESO [ (df_PROCESO['AÑO'].isin([year_selection])) & (df_PROCESO['UNIDAD NEGOCIO'].isin(fundo1)) & (df_PROCESO['VARIEDAD'].isin(VARIEDAD2)) ]
    
    number_of_result = df_PROCESO.shape[0]
    st.markdown(f'*Available Result: {number_of_result}*')

    unpivot_PROCESO_gral = pd.melt(df_PROCESO, 
    id_vars=['UNIDAD NEGOCIO','SEMANA', 'AÑO', 'FECHAPROD', 'FECHACOSE', 'FECHARECEP', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'LOTE'],
    value_vars=['PESO NETO', 'EXPORTABLE', 'TOTAL SPM', 'TOTAL MN', 'MERMA TOTAL'], 
    var_name='TIPO PESO', value_name= 'PESO')

    #pivot de datos global y detalles
    unpivot_globalGRAFICO = pd.melt(df_PROCESO2122, 
    id_vars=['UNIDAD NEGOCIO','SEMANA', 'AÑO', 'FECHAPROD', 'FECHACOSE', 'FECHARECEP', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'LOTE'],
    value_vars=['PESO NETO', 'EXPORTABLE', 'TOTAL SPM', 'TOTAL MN', 'MERMA TOTAL'], 
    var_name='TIPO PESO', value_name= 'PESO')

    unpivot_globalGRAFICOdet = pd.melt(df_PROCESO2122,
    id_vars=['UNIDAD NEGOCIO','SEMANA', 'AÑO', 'FECHAPROD', 'FECHACOSE', 'FECHARECEP', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'LOTE'],
    value_vars=['MN 3A+', 'MN 1A2', 'EXTRA', 'PRIM', 'SEG Y TERC', 'SPM1', 'SPM2',
    'PRECALIBRE', 'DESECHO', 'DESHIDRA', 'SOBREPESO'], 
    var_name='TIPO PESO', value_name= 'PESO')

    # df_PROCESO
    # unpivot_PROCESO_gral
    grp_CAJAS = df_PROCESO.groupby(['UNIDAD NEGOCIO', 'AÑO', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'LOTE'])['TOTAL CAJAS'].sum().reset_index()

    up_PROCESOgrouped = unpivot_PROCESO_gral.groupby(['UNIDAD NEGOCIO', 'AÑO', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'LOTE', 'TIPO PESO'])['PESO'].sum().reset_index()

    up_PROCESOgroupedGLOBAL = unpivot_PROCESO_gral.groupby(['UNIDAD NEGOCIO', 'AÑO', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'TIPO PESO'])['PESO'].sum().reset_index()

    #agrupacion de datos global y detalles
    up_PROCESOgrouped2122 = unpivot_globalGRAFICO.groupby(['UNIDAD NEGOCIO', 'AÑO', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'TIPO PESO'])['PESO'].sum().reset_index()

    up_PROCESOgrouped2122det = unpivot_globalGRAFICOdet.groupby(['UNIDAD NEGOCIO', 'AÑO', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'CULTIVO', 'VARIEDAD', 'TIPO PESO'])['PESO'].sum().reset_index()

    #peso procesado por dia total
    up_PROCESOgroupedTIMESERIES = unpivot_globalGRAFICO.groupby(['AÑO', 'FECHAPROD', 'CLIENTE', 'PRODUCTOR', 
    'SERVICIO', 'TIPO PESO'])['PESO'].sum().reset_index()
    
    df_TimeseriesNeto = up_PROCESOgroupedTIMESERIES[(up_PROCESOgroupedTIMESERIES['TIPO PESO']=='PESO NETO')]

    df_2021 = up_PROCESOgrouped2122[(up_PROCESOgrouped2122['AÑO']==2021)]
    df_2022 = up_PROCESOgrouped2122[(up_PROCESOgrouped2122['AÑO']==2022)]
    df_2021det = up_PROCESOgrouped2122det[(up_PROCESOgrouped2122det['AÑO']==2021)]
    df_2022det = up_PROCESOgrouped2122det[(up_PROCESOgrouped2122det['AÑO']==2022)]

    SPM =['SPM1', 'SPM2']
    MN = ['MN 3A+', 'MN 1A2', 'EXTRA', 'PRIM', 'SEG Y TERC']
    MERMA = ['PRECALIBRE', 'DESECHO', 'DESHIDRA', 'SOBREPESO']

    #filtro de los conceptos de super mercado
    df_SPM2022 = df_2022det[ df_2022det['TIPO PESO'].isin(SPM) ]
    df_SPM2021 = df_2021det[ df_2021det['TIPO PESO'].isin(SPM) ]

    #filtro de los conceptos de mercado nacional
    df_MN2022 = df_2022det[ df_2022det['TIPO PESO'].isin(MN) ]
    df_MN2021 = df_2021det[ df_2021det['TIPO PESO'].isin(MN) ]

    #filtro de los conceptos de merma
    df_MERMA2022 = df_2022det[ df_2022det['TIPO PESO'].isin(MERMA) ]
    df_MERMA2021 = df_2021det[ df_2021det['TIPO PESO'].isin(MERMA) ]

    #contruccion de la data de porcentajes    
    df_percentage = up_PROCESOgrouped [ up_PROCESOgrouped['TIPO PESO'] != 'PESO NETO' ]
    # Using DataFrame.transform() method.
    up_PROCESOgrouped['%'] = 100 * df_percentage['PESO'] / df_percentage.groupby('LOTE')['PESO'].transform('sum')


    #peso exportable por dia total
    df_TimeseriesExportable = df_PROCESO2122.groupby(['AÑO', 'CLIENTE', 'PRODUCTOR', 'FECHAPROD',
    'SERVICIO', 'CULTIVO', 'VARIEDAD'])['% EXPO'].mean().reset_index()
    df_TimeseriesExportable['%'] = df_TimeseriesExportable['% EXPO'].round(decimals = 2) * 100
    df_TimeseriesExportable['%'] = df_TimeseriesExportable['%'].round(decimals=2)
    #df_TimeseriesExportable
    # df_TimeseriesExportable.sort_values(by=['%'])
    # .apply(lambda x: "{:.0f}".format((x)))
    
    # .apply(lambda x: round(x, 2))
    # .apply(lambda x: "${:.1f}k".format((x/1000)))
    #df_TimeseriesExportable.style.format('{:,.2f}%')

    #expander pesos
    expander_bar1 = st.expander("Distribución del Proceso por Consumidor")

    # Render
    if df_HEADER.shape[0]:
        pivot_PESOS = pd.pivot_table(df_percentage, index=['VARIEDAD', 'TIPO PESO'], columns='LOTE', values='PESO', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
        expander_bar1.dataframe(pivot_PESOS.style\
            .background_gradient(axis=1, cmap='BuGn', low=0, high=1.0)\
            .applymap(highlight_max).format('{:,.2f}')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
        )
    
    #expander exportable
    expander_bar2 = st.expander("Distribución % del Proceso por Consumidor")

    # Render
    if df_HEADER.shape[0]:
        pivot_PORCENTAJE = pd.pivot_table(up_PROCESOgrouped, index=['VARIEDAD', 'TIPO PESO'], columns='LOTE', values='%', fill_value=np.nan)
        expander_bar2.dataframe(pivot_PORCENTAJE.style\
            .background_gradient(axis=1, cmap='BuGn', low=0, high=1.0)\
            .applymap(highlight_max).format('{:,.2f}%')\
            .applymap(lambda x: color_nan_white(x))\
            .applymap(lambda x: color_nan_white_background(x))
        )

    st.markdown('---')
    ### --- COLUMNS ---
    col2, col3 = st.columns((1.3,1))

    with col2:
        st.caption('Distribución de KG Procesados por Consumidor')

        fig = px.bar(up_PROCESOgrouped.sort_values(by=['LOTE'], ascending=False), y='LOTE', x='PESO', template="simple_white",
        color ='TIPO PESO', barmode='stack', height=350, width=600)
        fig.update_traces(textfont_size=15, textangle=0, 
        textposition="inside", texttemplate = "%{value:,s}",)
        fig.update_layout(title_text='KG PROCESADOS',
        legend=dict(
            title=None, orientation="v", y=1, x=1, 
            font=dict(
                family="Segoe UI Symbol",
                size=11.5,
                color="black"
                ),
            borderwidth=0
            )
        )
        st.plotly_chart(fig)

    with col3:
        st.caption('Distribución de Cajas Procesadas por Consumidor')

        fig = px.bar(grp_CAJAS.sort_values(by=['LOTE']), y='LOTE', x='TOTAL CAJAS', template="simple_white",
        color ='LOTE', barmode='stack', height=350, width=500)
        fig.update_traces(textfont_size=15, textangle=0, 
        textposition="inside", texttemplate = "%{value:,s}",)
        fig.update_layout(
        legend=dict(
            title=None, orientation="v", y=1, x=1, 
            font=dict(
                family="Segoe UI Symbol",
                size=11.5,
                color="black"
                ),
            borderwidth=0
            )
        )
        fig.update_layout(title_text='TOTAL DE CAJAS', yaxis={'visible': False, 'showticklabels': False})
        st.plotly_chart(fig)

    st.markdown('---')

    #grafico de % de Pesos por año
    st.caption('Distribución % del Proceso 2022 vs 2021')

    fig = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2022', '', '2021'])
    fig.add_trace(go.Pie(labels=df_2022['TIPO PESO'], values=df_2022['PESO'], scalegroup='one',
                        name="PROCESO 2022", hole=.5), 1, 1)
    fig.add_trace(go.Pie(values=[''], scalegroup='one',
                        hole=.5), 1, 2)
    fig.add_trace(go.Pie(labels=df_2021['TIPO PESO'], values=df_2021['PESO'], scalegroup='one',
                        name="PROCESO 2021", hole=.5), 1, 3)
    fig.update_layout(title_text='PROCESO', autosize=False, width=1000, height=400)
    st.plotly_chart(fig)

    st.markdown('---')

    st.caption('Distribución % de Super Mercado 2022 vs 2021')
    fig2 = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2022', '', '2021'])
    fig2.add_trace(go.Pie(labels=df_SPM2022['TIPO PESO'], values=df_SPM2022['PESO'], scalegroup='one',
                        name="SPM 2022", hole=.5), 1, 1)
    fig2.add_trace(go.Pie(values=[''], scalegroup='one',
                        hole=.5), 1, 2)                        
    fig2.add_trace(go.Pie(labels=df_SPM2021['TIPO PESO'], values=df_SPM2021['PESO'], scalegroup='one',
                        name="SPM 2021", hole=.5), 1, 3)
    fig2.update_layout(title_text='SUPER MERCADO', autosize=False, width=1000, height=400)
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig2.update_traces(marker=dict(colors=colors))
    st.plotly_chart(fig2)

    st.markdown('---')

    st.caption('Distribución % de Mercado Nacional 2022 vs 2021')
    fig2 = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2022', '', '2021'])
    fig2.add_trace(go.Pie(labels=df_MN2022['TIPO PESO'], values=df_MN2022['PESO'], scalegroup='one',
                        name="MN 2022", hole=.5), 1, 1)
    fig2.add_trace(go.Pie(values=[''], scalegroup='one',
                        hole=.5), 1, 2)                        
    fig2.add_trace(go.Pie(labels=df_MN2021['TIPO PESO'], values=df_MN2021['PESO'], scalegroup='one',
                        name="MN 2021", hole=.5), 1, 3)
    fig2.update_layout(title_text='MERCADO NACIONAL', autosize=False, width=1000, height=400)
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig2.update_traces(marker=dict(colors=colors))
    st.plotly_chart(fig2)

    st.markdown('---')

    st.caption('Distribución % de la Merma 2022 vs 2021')
    fig2 = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2022', '', '2021'])
    fig2.add_trace(go.Pie(labels=df_MERMA2022['TIPO PESO'], values=df_MERMA2022['PESO'], scalegroup='one',
                        name="MERMA 2022", hole=.5), 1, 1)
    fig2.add_trace(go.Pie(values=[''], scalegroup='one',
                        hole=.5), 1, 2)                        
    fig2.add_trace(go.Pie(labels=df_MERMA2021['TIPO PESO'], values=df_MERMA2021['PESO'], scalegroup='one',
                        name="MERMA 2021", hole=.5), 1, 3)
    fig2.update_layout(title_text='MERMA', autosize=False, width=1000, height=400)
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig2.update_traces(marker=dict(colors=colors))
    st.plotly_chart(fig2)

    st.markdown('---')

    #grafico de % de Pesos por año
    st.caption('Time series - Peso neto procesado')
    #timeseries pico de proceso
    figdate = px.line(df_TimeseriesNeto, x='FECHAPROD', y='PESO', title='Peso neto procesado MMPP - [Propio]', width=1000, template="ggplot2")
    figdate.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=14,label="2w",step="day",stepmode="todate"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    st.plotly_chart(figdate)

    st.markdown('---')

    #grafico de % de Pesos por año
    st.caption('Time series - Peso exportable')
    #timeseries pico de proceso
    figdate = px.line(df_TimeseriesExportable, x='FECHAPROD', y='%', title='Peso exportable MMPP - [Propio]', width=1000, template="ggplot2", text="%")
    figdate.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=14,label="2w",step="day",stepmode="todate"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    # figdate.update_xaxes(
    # range=[df_TimeseriesExportable.index[0], df_TimeseriesExportable.index[-1]],
    # rangeslider_range=[df_TimeseriesExportable.index[0], df_TimeseriesExportable.index[-1]]
    # )
    figdate.update_traces(textposition="bottom right")
    st.plotly_chart(figdate)

elif selected == "Schedule":

    df_PROYECCION['FECHAPROG'] = df_PROYECCION['FECHA PROGRAMADA'].apply(lambda x: datetime.strftime(x,"%m.%d"))
    df_PROYECCION['FECHAPROY'] = df_PROYECCION['FECHAPROYECCION'].apply(lambda x: datetime.strftime(x,"%d/%m"))
    
    df_COSECHA['FECHAPROG'] = df_COSECHA['fecha'].apply(lambda x: datetime.strftime(x,"%m.%d"))
    df_COSECHA['PESO'] = df_COSECHA['peso_neto']/1000
    df_COSECHA = df_COSECHA.round({'PESO': 0})
    
    df_PROYECCION = df_PROYECCION.round({'PESO': 0})

    office = df_PROYECCION['UNIDAD NEGOCIO'].unique().tolist()
    variety = df_PROYECCION['VARIEDAD'].unique().tolist()
    
    #slider semana
    office_selection = st.multiselect('Fundo:',
                                    office,
                                    default=office)

    variety_selection = st.multiselect('Variedad:',
                                    variety,
                                    default= None)

    date = st.date_input("Seleccionar rango de fechas", [])

    if len(date) > 1:
        startdate = date[0]#.strftime("%d %b")
        enddate = date[1]#.strftime("%d %b")
        
        dtPROY_filterdate = df_PROYECCION.set_index('FECHA PROGRAMADA')[startdate:enddate]
        dtCOSECHA_filterdate = df_COSECHA.set_index('fecha')[startdate:enddate]
        
        #data proyeccion histórico
        mask = dtPROY_filterdate[ (dtPROY_filterdate['UNIDAD NEGOCIO'].isin(office_selection)) & (dtPROY_filterdate['VARIEDAD'].isin(variety_selection)) ]

        #data cosecha real
        mask2 = dtCOSECHA_filterdate[ (dtCOSECHA_filterdate['UNIDAD NEGOCIO'].isin(office_selection)) & (dtCOSECHA_filterdate['VARIEDAD'].isin(variety_selection)) ]

        number_of_result = mask.shape[0]
        st.markdown(f'*Available Result: {number_of_result}*')



        #tab3 Reporte general de Cosecha kg
        st.caption('Histórico de proyección de Cosecha vs Cosecha REAL')

        if len(office_selection) == 1:
            df_PROYgrouped = mask.groupby(['UNIDAD NEGOCIO','VARIEDAD','FECHAPROG','FECHAPROY','FECHAPROYECCION'])['PESO'].sum().reset_index()
            df_COSEgrouped = mask2.groupby(['UNIDAD NEGOCIO','VARIEDAD','FECHAPROG'])['PESO'].sum().reset_index()
        elif len(office_selection) >  1:
            df_PROYgrouped = mask.groupby(['VARIEDAD','FECHAPROG','FECHAPROY','FECHAPROYECCION'])['PESO'].sum().reset_index()
            df_COSEgrouped = mask2.groupby(['VARIEDAD','FECHAPROG'])['PESO'].sum().reset_index()

    # pivot_PROYECCION = pd.pivot_table(df_PROYECCION[mask].sort_values(by="FECHAPROYECCION"), index=['VARIEDAD','LOTE','FECHAPROYECCION'], columns=['FECHA PROGRAMADA'], values='PESO', fill_value=np.nan, margins=True, aggfunc=sum, margins_name='Total')
    # variety = df_PROYECCION['VARIEDAD'].unique().tolist()
    #lotes = dt_filterdate[mask]['LOTE'].unique().tolist()

        if mask.shape[0]:
            df_PROYgrouped.sort_values(by=['FECHAPROG'])
            df_COSEgrouped.sort_values(by=['FECHAPROG'])
            LOTE='LOTES'

            fig = px.bar(
            df_PROYgrouped.sort_values('FECHAPROG'), #el x en este caso ordenado por fecha programada
            x='FECHAPROG', y='PESO', color='VARIEDAD', #lo mismo con la legend de la variedad
            animation_frame='FECHAPROY', #la barra de la animacion en este caso la fecha proyectada
            animation_group='FECHAPROG', #la fecha de las barras = x
            hover_name='VARIEDAD', #de la mano del color
            category_orders={'FECHAPROY':['16/06','23/06','30/06','07/07','13/07','21/07','27/07','01/08','06/08','11/08','13/08','18/08','27/08', '31/08']},
            range_y=[0,400],
            range_x=[-1,12],
            height=500,
            width=1000,
            text='PESO'
            )

            #fig.update_traces(textfont_size=7, textangle=0, textposition="auto")
            fig.update_layout(
                legend=dict(
                title=None, 
                borderwidth=0
                ),
                #autosize=True,
                template='simple_white',
                margin=dict(r=10, t=25, b=40, l=10)
                #transition = {'duration': 1000}            
                )
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
            fig.update_traces(textposition='auto', marker_color="#f0c32b")
            fig.update_xaxes(automargin=True)

            #add trace cosecha real
            fig2 = px.scatter(
                df_COSEgrouped.sort_values('FECHAPROG'),
                x='FECHAPROG',
                y='PESO',
                # x=['08.01','08.02','08.03','08.04','08.05','08.06'],
                # y=[80,100,110,90,120,130],
                color_discrete_sequence=['black'],
                labels=dict(FECHAPROG="FECHA REAL", PESO="PESO (tn)", VARIEDAD="VARIEDAD")
                # x=df_PROYgrouped['FECHAPROG'],
                # y=df_PROYgrouped['PESO']
                )
            fig2.update_traces(mode="lines+markers", name='TN reales')
            fig2.update_traces(line_color='black', line_width=2)
            fig2.update_traces(selector=-1, showlegend=True)
            ####################

            fig.add_traces(fig2._data)
            #fig.update_layout(hovermode="x unified")
            
            st.plotly_chart(fig)
            

elif selected == "COMEX":

    #filter year
    year2122 = list(reversed(range(2021,2023)))
    year_selection = st.selectbox('Año:', year2122)

    #mask = (df_PROCESO['SEMANA'].between(*semana_selection)) & (df_PROCESO['UNIDAD NEGOCIO'].isin(office_selection)) & (df_PROCESO['VARIEDAD'].isin(variety_selection))
    #df_PROCESO = df_PROCESO[ (df_COSECHA['LOTE'].isin(df_MO['CONSUMIDOR'])) & (df_COSECHA['VARIEDAD'].isin(df_MO['VARIEDAD'])) ]
    df_DESPACHOS2122 = df_DESPACHOS [ (df_DESPACHOS['AÑO'].isin(year2122)) & (df_DESPACHOS['UNIDAD NEGOCIO'].isin(fundo1)) & (df_DESPACHOS['VARIEDAD'].isin(VARIEDAD2)) ]
    df_DESPACHOS = df_DESPACHOS [ (df_DESPACHOS['AÑO'].isin([year_selection])) & (df_DESPACHOS['UNIDAD NEGOCIO'].isin(fundo1)) & (df_DESPACHOS['VARIEDAD'].isin(VARIEDAD2)) ]

    number_of_result = df_DESPACHOS.shape[0]
    st.markdown(f'*Available Result: {number_of_result}*')

    #agrupador total de cajas por cliente packing
    grp_CAJASCLILOTE = df_DESPACHOS.groupby(['AÑO', 'CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 
    'LOTE'])['CAJAS'].sum().reset_index()

    #agrupador total de cajas por cliente packing
    grp_CAJAStotalCLI = df_DESPACHOS.groupby(['AÑO','CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR'])['CAJAS'].sum().reset_index()
    grp_CAJAStotal = df_DESPACHOS.groupby(['AÑO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR'])['CAJAS'].sum().reset_index()
    grp_CATtotal = df_DESPACHOS.groupby(['AÑO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'CATEGORIA'])['CAJAS'].sum().reset_index()

    grp_CAJAStotalCLI2122 = df_DESPACHOS2122.groupby(['AÑO','CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR'])['CAJAS'].sum().reset_index()

    #grp_FCLtotalCLI2122
    grp_CAT2122 = df_DESPACHOS2122.groupby(['AÑO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'CATEGORIA'])['CAJAS'].sum().reset_index()
    grp_CAL2122 = df_DESPACHOS2122.groupby(['AÑO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'CALIBRE'])['CAJAS'].sum().reset_index()
    grp_ENV2122 = df_DESPACHOS2122.groupby(['AÑO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'ENVASE'])['CAJAS'].sum().reset_index()

    grp_CAT2122_CLI = df_DESPACHOS2122.groupby(['AÑO', 'CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'CATEGORIA'])['CAJAS'].sum().reset_index()
    grp_CAL2122_CLI = df_DESPACHOS2122.groupby(['AÑO', 'CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'CALIBRE'])['CAJAS'].sum().reset_index()
    grp_ENV2122_CLI = df_DESPACHOS2122.groupby(['AÑO', 'CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 'ENVASE'])['CAJAS'].sum().reset_index()
    #agrupador total de cajas por cliente packing por lote
    grp_CAJASCLIlote = df_DESPACHOS.groupby(['AÑO','CLIENTE PACKING HOMO', 'CULTIVO', 'VARIEDAD', 'CLIENTE', 'PRODUCTOR', 
    'LOTE', 'PACKINGLIST', 'CONTENEDOR', 'PALETA', 'CATEGORIA', 'CALIBRE'])['CAJAS'].sum().reset_index()


    df_2021 = grp_CAJAStotalCLI2122[(grp_CAJAStotalCLI2122['AÑO']==2021)]
    df_2022 = grp_CAJAStotalCLI2122[(grp_CAJAStotalCLI2122['AÑO']==2022)]

    df_CAT2021 = grp_CAT2122[(grp_CAT2122['AÑO']==2021)]
    df_CAT2022 = grp_CAT2122[(grp_CAT2122['AÑO']==2022)]
    df_CAL2021 = grp_CAL2122[(grp_CAL2122['AÑO']==2021)]
    df_CAL2022 = grp_CAL2122[(grp_CAL2122['AÑO']==2022)]
    df_ENV2021 = grp_ENV2122[(grp_ENV2122['AÑO']==2021)]
    df_ENV2022 = grp_ENV2122[(grp_ENV2122['AÑO']==2022)]

    df_CAT2021_CLI = grp_CAT2122_CLI[(grp_CAT2122_CLI['AÑO']==2021)]
    df_CAT2022_CLI = grp_CAT2122_CLI[(grp_CAT2122_CLI['AÑO']==2022)]
    df_CAL2021_CLI = grp_CAL2122_CLI[(grp_CAL2122_CLI['AÑO']==2021)]
    df_CAL2022_CLI = grp_CAL2122_CLI[(grp_CAL2122_CLI['AÑO']==2022)]
    df_ENV2021_CLI = grp_ENV2122_CLI[(grp_ENV2122_CLI['AÑO']==2021)]
    df_ENV2022_CLI = grp_ENV2122_CLI[(grp_ENV2122_CLI['AÑO']==2022)]

    st.markdown('---')

    st.caption('Cajas despachadas por Cliente [EXPO]')

    fig = px.bar(grp_CAJAStotalCLI.sort_values(by=['CAJAS'], ascending=True), x='CAJAS', y='CLIENTE PACKING HOMO', template="simple_white",
    color ='VARIEDAD', barmode='stack', height=500, width=1000)
    fig.update_traces(textfont_size=15, textangle=0, 
    textposition="inside", texttemplate = "%{value:,s}",)
    fig.update_layout(title_text='TOTAL DE CAJAS [EXPO]',
    legend=dict(
        title=None, orientation="v", y=1, x=1, 
        font=dict(
            family="Segoe UI Symbol",
            size=11.5,
            color="black"
            ),
        borderwidth=0
        )
    )
    st.plotly_chart(fig)


    #expander grafico
    expander_bar1 = st.expander("Cajas despachadas por Lote [EXPO]")

    fig = px.bar(grp_CAJASCLILOTE.sort_values(by=['CAJAS'], ascending=True), x='CAJAS', y='CLIENTE PACKING HOMO', text='LOTE', template="simple_white",
    color ='VARIEDAD', barmode='stack', height=500, width=980)
    fig.update_traces(textfont_size=15, textangle=0, 
    textposition="inside", texttemplate = "%{value:,s}",)
    fig.update_layout(title_text='TOTAL DE CAJAS POR LOTE [EXPO]',
    legend=dict(
        title=None, orientation="v", y=1, x=1, 
        font=dict(
            family="Segoe UI Symbol",
            size=11.5,
            color="black"
            ),
        borderwidth=0
        )
    )
    expander_bar1.plotly_chart(fig)

    st.markdown('---')

    #grafico de % de Pesos por año
    col1, col2 = st.columns((1,1))

    with col1:
        st.caption('Distribución de Cajas Exportadas por Variedad')

        fig = px.bar(grp_CAJAStotal.sort_values(by=['VARIEDAD'], ascending=False), x='VARIEDAD', y='CAJAS', template="simple_white",
        color ='VARIEDAD', barmode='stack', height=350, width=500)
        fig.update_traces(textfont_size=15, textangle=0, 
        textposition="inside", texttemplate = "%{value:,s}",)
        fig.update_layout(title_text='CAJAS POR VARIEDAD',
        legend=dict(
            title=None, orientation="v", y=1, x=1, 
            font=dict(
                family="Segoe UI Symbol",
                size=11.5,
                color="black"
                ),
            borderwidth=0
            )
        )
        colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
        st.plotly_chart(fig)

    with col2:
        st.caption('Distribución de Categoria por Variedad')

        fig = make_subplots(1, 1, specs=[[{'type':'domain'}]])
        fig.add_trace(go.Pie(labels=grp_CATtotal['CATEGORIA'], values=grp_CATtotal['CAJAS']), 1, 1)
        fig.update_layout(title_text='CATEGORIA POR VARIEDAD', height=355)
        #fig = px.pie(labels=grp_CATtotal['CATEGORIA'], values=grp_CATtotal['CAJAS'], height=350, width=500)
        colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
        st.plotly_chart(fig)

    st.markdown('---')


    #checkbox filtro por cliente
    agree = st.checkbox('por Cliente EXPO')

    if agree:
        #slider cliente packing
        clipacking = df_DESPACHOS['CLIENTE PACKING HOMO'].unique().tolist()
        
        #slider semana
        clipacking_selection = st.selectbox('Cliente EXPO:', clipacking)

        if clipacking_selection[0]:

            df_CAT2022_CLI = df_CAT2022_CLI [ (df_CAT2022_CLI['CLIENTE PACKING HOMO'].isin([clipacking_selection])) ]
            df_CAT2021_CLI = df_CAT2021_CLI [ (df_CAT2021_CLI['CLIENTE PACKING HOMO'].isin([clipacking_selection])) ]
            #categoria pie chart
            st.caption('Distribución % de Categoría 2022 vs 2021')
            fig = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
                                subplot_titles=['2022', '', '2021'])
            fig.add_trace(go.Pie(labels=df_CAT2022_CLI['CATEGORIA'], values=df_CAT2022_CLI['CAJAS'], scalegroup='one',
                                name="CATEGORIA 2022", hole=.5), 1, 1)
            fig.add_trace(go.Pie(values=[''], scalegroup='one',
                                hole=.5), 1, 2)
            fig.add_trace(go.Pie(labels=df_CAT2021_CLI['CATEGORIA'], values=df_CAT2021_CLI['CAJAS'], scalegroup='one',
                                name="CATEGORIA 2021", hole=.5), 1, 3)
            colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
            fig.update_layout(title_text='PROCESO', autosize=False, width=1000, height=400)
            st.plotly_chart(fig)

            st.markdown('---')

            # #calibre pie chart
            # st.caption('Distribución % de Calibre 2022 vs 2021')
            # fig = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
            #                     subplot_titles=['2022', '', '2021'])
            # fig.add_trace(go.Pie(labels=df_CAL2022['CALIBRE'], values=df_CAL2022['CAJAS'], scalegroup='one',
            #                     name="CALIBRE 2022", hole=.5), 1, 1)
            # fig.add_trace(go.Pie(values=[''], scalegroup='one',
            #                     hole=.5), 1, 2)
            # fig.add_trace(go.Pie(labels=df_CAL2021['CALIBRE'], values=df_CAL2021['CAJAS'], scalegroup='one',
            #                     name="CALIBRE 2021", hole=.5), 1, 3)
            # colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
            # fig.update_layout(title_text='PROCESO', autosize=False, width=1000, height=400)
            # st.plotly_chart(fig)

            # st.markdown('---')

            # #presentación pie chart
            # st.caption('Distribución % de la Presentación 2022 vs 2021')
            # fig = make_subplots(1, 3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]],
            #                     subplot_titles=['2022', '', '2021'])
            # fig.add_trace(go.Pie(labels=df_ENV2022['ENVASE'], values=df_ENV2022['CAJAS'], scalegroup='one',
            #                     name="PRESENTACION 2022", hole=.5), 1, 1)
            # fig.add_trace(go.Pie(values=[''], scalegroup='one',
            #                     hole=.5), 1, 2)
            # fig.add_trace(go.Pie(labels=df_ENV2021['ENVASE'], values=df_ENV2021['CAJAS'], scalegroup='one',
            #                     name="PRESENTACION 2021", hole=.5), 1, 3)
            # colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
            # fig.update_layout(title_text='PROCESO', autosize=False, width=1000, height=400)
            # st.plotly_chart(fig)
