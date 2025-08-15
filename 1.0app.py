import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Dashboard Marketing Digital", page_icon="💡", layout="wide")

# ---------- Helpers para limpar números/monetário ----------
def clean_money_series(series):
    s = series.fillna('').astype(str)
    s = s.str.replace(r'[R$\s]', '', regex=True)  # remove R$, espaços
    # Se tiver ambos '.' e ',' -> assume '.' separador de milhares e ',' decimal
    both = s.str.contains(r'\.') & s.str.contains(r',')
    s.loc[both] = s.loc[both].str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    # Se tiver apenas ',' -> decimal comma
    only_comma = s.str.contains(',') & ~s.str.contains(r'\.')
    s.loc[only_comma] = s.loc[only_comma].str.replace(',', '.', regex=False)
    # Remover quaisquer caracteres que não sejam dígito, ponto, ou sinal de menos
    s = s.str.replace(r'[^\d\.\-]', '', regex=True)
    return pd.to_numeric(s, errors='coerce').fillna(0)

def ensure_numeric_series(df, col):
    if col not in df.columns:
        return pd.Series(0, index=df.index)
    return clean_money_series(df[col])

# ---------- Leitura das fontes (Google + Meta) com marcação da origem ----------
def ler_dados_google(sheet_id, gids):
    tabelas = []
    for gid in gids:
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(url)
        df.columns = df.columns.str.strip()
        df['source'] = 'google'
        # Normalizar colunas que podem existir
        # Limpeza numérica será feita depois com helpers
        tabelas.append(df)
    return pd.concat(tabelas, ignore_index=True) if tabelas else pd.DataFrame()

def ler_dados_meta(url):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df['source'] = 'meta'
    # Garantir Day como datetime se existir
    if 'Day' in df.columns:
        df['Day'] = pd.to_datetime(df['Day'], errors='coerce')
    return df

# ---------- URLs / IDs (suas) ----------
sheet_id = "1FPp-nlswMeyVY1nA9lEwnEkephShEVB2_syDTFEwU3k"
gids_google = ["315709897"]
url_meta = "https://docs.google.com/spreadsheets/d/1FPp-nlswMeyVY1nA9lEwnEkephShEVB2_syDTFEwU3k/export?format=csv&gid=911679538"

# ---------- Ler dados ----------
df_google = ler_dados_google(sheet_id, gids_google)
df_meta = ler_dados_meta(url_meta)

# concat (mantendo origem)
df_unificado = pd.concat([df_google, df_meta], ignore_index=True)

# ---------- Garantir Day e limpeza numérica geral ----------
df_unificado['Day'] = pd.to_datetime(df_unificado.get('Day'), errors='coerce')

# Colunas que usaremos: Vendas, Revenue, Amount Spent, Cost Spend / Cost (Spend), Clicks, Impressions
df_unificado['Vendas'] = ensure_numeric_series(df_unificado, 'Vendas')
df_unificado['Revenue'] = ensure_numeric_series(df_unificado, 'Revenue')
df_unificado['Amount_Spent_Clean'] = ensure_numeric_series(df_unificado, 'Amount Spent')

# Para Google: aceitar 'Cost Spend' ou 'Cost (Spend)'
if 'Cost Spend' in df_unificado.columns:
    df_unificado['Cost_Spend_Clean'] = ensure_numeric_series(df_unificado, 'Cost Spend')
elif 'Cost (Spend)' in df_unificado.columns:
    df_unificado['Cost_Spend_Clean'] = ensure_numeric_series(df_unificado, 'Cost (Spend)')
else:
    df_unificado['Cost_Spend_Clean'] = 0

# Clicks e Impressions para métricas
df_unificado['Clicks'] = ensure_numeric_series(df_unificado, 'Clicks')
df_unificado['Impressions'] = ensure_numeric_series(df_unificado, 'Impressions')

# ---------- Investimento por linha: se source==meta use Amount_Spent_Clean, se google use Cost_Spend_Clean ----------
# Se por acaso existir linhas sem 'source' ou com outros sources, pega a soma (fallback)
def calcular_investimento_row(r):
    if r.get('source') == 'meta':
        return r.get('Amount_Spent_Clean', 0)
    if r.get('source') == 'google':
        return r.get('Cost_Spend_Clean', 0)
    # fallback: soma dos dois (caso alguma linha já venha inteira)
    return r.get('Amount_Spent_Clean', 0) + r.get('Cost_Spend_Clean', 0)

df_unificado['Investimento'] = df_unificado.apply(calcular_investimento_row, axis=1)

# ---------- Agora os filtros (lado esquerdo da tela) - usando Campaign Name se existir ----------
st.sidebar.header("🔍 Filtros")

min_data = df_unificado['Day'].min()
max_data = df_unificado['Day'].max()

st.sidebar.subheader("Período Atual")
periodo_atual = st.sidebar.date_input(
    "Selecione o período atual",
    [min_data, max_data],
    min_value=min_data,
    max_value=max_data,
    key="periodo_atual"
)

st.sidebar.subheader("Período Comparativo")
periodo_comp = st.sidebar.date_input(
    "Selecione o período comparativo",
    [min_data - timedelta(days=7), min_data - timedelta(days=1)],
    min_value=min_data - timedelta(days=365),
    max_value=max_data,
    key="periodo_comp"
)

def ajustar_datas(periodo):
    if isinstance(periodo, (tuple, list)):
        return periodo[0], periodo[1]
    return periodo, periodo

atual_start, atual_end = ajustar_datas(periodo_atual)
comp_start, comp_end = ajustar_datas(periodo_comp)

campanhas = sorted(df_unificado.get('Campaign Name', pd.Series([], dtype=object)).dropna().unique())
campanhas_selecionadas = st.sidebar.multiselect('Selecione a(s) campanha(s)', campanhas, default=campanhas)


# Aplicar filtros (usa df_unificado já limpo)
cond_campanha = df_unificado.get('Campaign Name').isin(campanhas_selecionadas) if 'Campaign Name' in df_unificado.columns else (pd.Series(True, index=df_unificado.index))
df_atual = df_unificado[
    cond_campanha &
    (df_unificado['Day'] >= pd.to_datetime(atual_start)) &
    (df_unificado['Day'] <= pd.to_datetime(atual_end))
].copy()

df_comp = df_unificado[
    cond_campanha &
    (df_unificado['Day'] >= pd.to_datetime(comp_start)) &
    (df_unificado['Day'] <= pd.to_datetime(comp_end))
].copy()

# ---------- Métricas (usar Investimento condicional) ----------
def calcular_metricas(df):
    impressoes = df['Impressions'].sum()
    cliques = df['Clicks'].sum()
    vendas = df['Vendas'].sum()
    receita = df['Revenue'].sum()
    investimento = df['Investimento'].sum()
    cpc = investimento / cliques if cliques != 0 else 0
    ctr = (cliques / impressoes) * 100 if impressoes != 0 else 0
    cpa = investimento / vendas if vendas != 0 else 0
    taxa_conv = (vendas / cliques) * 100 if cliques != 0 else 0
    roas = receita / investimento if investimento != 0 else 0
    return {
        "Investimento": investimento,
        "Impressões": impressoes,
        "Cliques": cliques,
        "CTR": ctr,
        "CPC": cpc,
        "Vendas": vendas,
        "CPA": cpa,
        "Taxa de Conversão": taxa_conv,
        "Revenue": receita,
        "ROAS": roas
    }

metricas_atual = calcular_metricas(df_atual)
metricas_comp = calcular_metricas(df_comp)

# ---------- Overview (mantive seu layout, usando métricas calculadas) ----------
def formatar_reais(valor):
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

st.title("Dashboard Marketing Digital")
st.write("OVERVIEW")

col1_m, col2_m, col3_m, col4_m, col5_m = st.columns(5)
col1_m.metric("Investimento", formatar_reais(metricas_atual["Investimento"]))
col1_m.markdown("", unsafe_allow_html=True)
col2_m.metric("Impressões", f"{metricas_atual['Impressões']:,}")
col3_m.metric("Cliques", f"{metricas_atual['Cliques']:,}")
col4_m.metric("CTR", f"{metricas_atual['CTR']:.2f}%")
col5_m.metric("CPC", formatar_reais(metricas_atual["CPC"]))

col6_m, col7_m, col8_m, col9_m, col10_m = st.columns(5)
col6_m.metric("Vendas", f"{metricas_atual['Vendas']:,}")
col7_m.metric("CPA", formatar_reais(metricas_atual["CPA"]))
col8_m.metric("Taxa de Conversão", f"{metricas_atual['Taxa de Conversão']:.2f}%")
col9_m.metric("Revenue", formatar_reais(metricas_atual["Revenue"]))
col10_m.metric("ROAS", f"{metricas_atual['ROAS']:.2f}")

st.markdown("---")

# ---------- Séries temporais filtradas (usando df_atual) ----------
# Agregar por dia para Vendas e Investimento -> CPA por dia
df_time = df_atual.groupby('Day', as_index=False).agg({'Vendas':'sum', 'Investimento':'sum'})
df_time['CPA'] = df_time.apply(lambda r: (r['Investimento'] / r['Vendas']) if r['Vendas']>0 else 0, axis=1)

# Agregar ROAS por dia
df_roas = df_atual.groupby('Day', as_index=False).agg({'Revenue':'sum', 'Investimento':'sum'})
df_roas['ROAS'] = df_roas.apply(lambda r: (r['Revenue'] / r['Investimento']) if r['Investimento']>0 else 0, axis=1)

# ---------- Gráficos (2 colunas) ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vendas e CPA")

    fig_vendas_cpa = go.Figure()

    # Área para Vendas (laranja escuro)
    fig_vendas_cpa.add_trace(go.Scatter(
        x=df_time['Day'],
        y=df_time['Vendas'],
        mode='lines',
        name='Vendas',
        line=dict(color='rgba(255,140,0,0)'),  # linha invisível
        fill='tozeroy',
        fillcolor='rgba(255,140,0,0.7)',  # laranja escuro com transparência
        yaxis='y1'
    ))

    # Linha para CPA (azul claro)
    fig_vendas_cpa.add_trace(go.Scatter(
        x=df_time['Day'],
        y=df_time['CPA'],
        mode='lines',
        name='CPA (R$)',
        line=dict(color='rgba(135,206,250,1)', width=3),  # azul claro contínua
        yaxis='y2'
    ))

    # Layout
    fig_vendas_cpa.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Dia'),
        yaxis=dict(showgrid=False, title='Vendas', showticklabels=False),      # escala vertical invisível
        yaxis2=dict(showgrid=False, title='CPA (R$)', overlaying='y', side='right', showticklabels=False),  # escala vertical invisível
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=40, t=30, b=10),
        hovermode='x unified'
    )

    st.plotly_chart(fig_vendas_cpa, use_container_width=True)

with col2:
    st.subheader("ROAS")
    fig_roas_plot = go.Figure()
    fig_roas_plot.add_trace(go.Scatter(
        x=df_roas['Day'],
        y=df_roas['ROAS'],
        mode='lines+markers',
        name='ROAS',
        line=dict(color='rgba(46,204,113,0.9)', width=3),
        marker=dict(size=6)
    ))
    fig_roas_plot.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Dia'),
        yaxis=dict(showgrid=False, title='ROAS'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified'
    )
    st.plotly_chart(fig_roas_plot, use_container_width=True)



# ========================
# Gráficos de Pizza por Canal

# URLs de export CSV corretas
# ========================
url_google = "https://docs.google.com/spreadsheets/d/1FPp-nlswMeyVY1nA9lEwnEkephShEVB2_syDTFEwU3k/export?format=csv&gid=315709897"
url_meta   = "https://docs.google.com/spreadsheets/d/1FPp-nlswMeyVY1nA9lEwnEkephShEVB2_syDTFEwU3k/export?format=csv&gid=911679538"

# ========================
# Função para ler e limpar dados
# ========================
def ler_dados_corrigido(url, tipo='google'):
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    
    # Investimento
    if tipo == 'google':
        df['Investimento'] = df.get('Cost (Spend)', 0)
    else:
        df['Investimento'] = df.get('Amount Spent', 0)
    
    # Limpar e converter para float
    df['Investimento'] = df['Investimento'].astype(str).str.replace('R\$', '', regex=True).str.replace(',', '.', regex=False)
    df['Investimento'] = pd.to_numeric(df['Investimento'], errors='coerce').fillna(0)
    
    # Revenue
    df['Revenue'] = df.get('Revenue', 0)
    df['Revenue'] = df['Revenue'].astype(str).str.replace('R\$', '', regex=True).str.replace(',', '.', regex=False)
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    
    # Vendas
    df['Vendas'] = pd.to_numeric(df.get('Vendas', 0), errors='coerce').fillna(0)
    
    # Canal
    if 'Advertising Channel' not in df.columns:
        df['Advertising Channel'] = 'outros'
    
    return df[['Advertising Channel', 'Investimento', 'Vendas', 'Revenue']]

# ========================
# Ler Google e Meta
# ========================
df_google = ler_dados_corrigido(url_google, 'google')
df_meta   = ler_dados_corrigido(url_meta, 'meta')

# ========================
# Criar DF unificado de canais
# ========================
df_canais = pd.concat([df_google, df_meta], ignore_index=True)

# ========================
# Padronizar canais
# ========================
def padronizar_canal(canal):
    canal = str(canal).lower()
    if "meta" in canal:
        return "meta"
    elif "display" in canal:
        return "display"
    elif "performance_max" in canal or "performance max" in canal or "pmax" in canal:
        return "performance_max"
    elif "search" in canal:
        return "search"
    else:
        return "outros"

df_canais['Canal Padronizado'] = df_canais['Advertising Channel'].apply(padronizar_canal)

# ========================
# Filtrar canais desejados
# ========================
canais_desejados = ["meta", "display", "performance_max", "search"]
df_canais = df_canais[df_canais['Canal Padronizado'].isin(canais_desejados)]

# ========================
# Agrupar canais e calcular métricas consolidadas
# ========================
df_canais_agg = df_canais.groupby('Canal Padronizado', as_index=False).agg({
    'Investimento': 'sum',
    'Revenue': 'sum',
    'Vendas': 'sum'
})

# Calcular ROAS corretamente (Revenue / Investimento)
df_canais_agg['ROAS'] = df_canais_agg.apply(lambda x: x['Revenue']/x['Investimento'] if x['Investimento']>0 else 0, axis=1)

# Garantir que todos os canais apareçam mesmo zerados
df_base = pd.DataFrame(canais_desejados, columns=['Canal Padronizado'])
df_canais_final = df_base.merge(df_canais_agg, on='Canal Padronizado', how='left').fillna(0)


# Gráficos de pizza na mesma linha


col1, col2, col3 = st.columns(3)

cores_canais = {
    "meta": "#FFA500",
    "display": "#FF7F50",
    "performance_max": "#FF4500",
    "search": "#FF1493"
}

# ---------- INVESTIMENTO ----------
fig_invest = px.pie(
    df_canais_final,
    names='Canal Padronizado',
    values='Investimento',
    title='Investimento por Canal',
    hole=0.4,
    color='Canal Padronizado',
    color_discrete_map=cores_canais
)
fig_invest.update_layout(paper_bgcolor='rgba(0,0,0,0)', title=dict(x=0.5))
fig_invest.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value:,.2f}')
col1.plotly_chart(fig_invest, use_container_width=True)

# ---------- VENDAS ----------
fig_vendas = px.pie(
    df_canais_final,
    names='Canal Padronizado',
    values='Vendas',
    title='Vendas por Canal',
    hole=0.4,
    color='Canal Padronizado',
    color_discrete_map=cores_canais
)
fig_vendas.update_layout(paper_bgcolor='rgba(0,0,0,0)', title=dict(x=0.5))
fig_vendas.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value:,.0f}')
col2.plotly_chart(fig_vendas, use_container_width=True)

# ---------- ROAS ----------
fig_roas = px.pie(
    df_canais_final,
    names='Canal Padronizado',
    values='ROAS',
    title='ROAS por Canal',
    hole=0.4,
    color='Canal Padronizado',
    color_discrete_map=cores_canais
)
fig_roas.update_layout(paper_bgcolor='rgba(0,0,0,0)', title=dict(x=0.5))
fig_roas.update_traces(textinfo='percent+label', hovertemplate='%{label}: %{value:,.2f}')
col3.plotly_chart(fig_roas, use_container_width=True)

# ========================
# Verificação do df_unificado
# ========================
if 'df_unificado' not in globals():
    st.error("Erro: df_unificado não está definido. Certifique-se de carregar os dados do dashboard antes deste gráfico.")
else:
    # ========================
    # Meses selecionados: Julho a Dezembro
    # ========================
    meses = ["Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"]
    mes_indices = [7, 8, 9, 10, 11, 12]  # meses correspondentes

    # Metas mensais para 2024 (Jul-Dez)
    metas_mensais = {
        "Julho": {"Vendas": 64, "Receita": 2300, "Investimento": 490},
        "Agosto": {"Vendas": 64, "Receita": 2310, "Investimento": 495},
        "Setembro": {"Vendas": 65, "Receita": 2320, "Investimento": 500},
        "Outubro": {"Vendas": 65, "Receita": 2300, "Investimento": 500},
        "Novembro": {"Vendas": 66, "Receita": 2350, "Investimento": 510},
        "Dezembro": {"Vendas": 70, "Receita": 2400, "Investimento": 520}
    }

    kpis = ["Vendas", "Receita", "Investimento"]

    # ========================
    # Garantir colunas numéricas
    # ========================
    for col in ['Vendas', 'Revenue', 'Amount Spent', 'Cost (Spend)']:
        if col in df_unificado.columns:
            df_unificado[col] = pd.to_numeric(df_unificado[col].fillna(0), errors='coerce')

    # Criar coluna única de investimento
    df_unificado['Investimento_Total'] = df_unificado.get('Amount Spent', 0) + df_unificado.get('Cost (Spend)', 0)

# ========================
# Preparar DataFrame com metas e realizados
# ========================
data_plot = []

for i, mes in enumerate(meses):
    for kpi in kpis:
        meta = metas_mensais[mes][kpi]

        # Filtrar mês específico, cuidando de valores nulos
        df_mes = df_unificado[df_unificado['Day'].dt.month == mes_indices[i]] if 'Day' in df_unificado.columns else pd.DataFrame()

        if kpi == "Vendas":
            realizado = df_mes['Vendas'].sum() if 'Vendas' in df_mes.columns else 0
        elif kpi == "Receita":
            realizado = df_mes['Revenue'].sum() if 'Revenue' in df_mes.columns else 0
        elif kpi == "Investimento":
            realizado = df_mes['Investimento_Total'].sum() if 'Investimento_Total' in df_mes.columns else 0

        data_plot.append({
            "Mês": mes,
            "KPI": kpi,
            "Meta": meta,
            "Realizado": realizado
        })

df_ano = pd.DataFrame(data_plot)

# ========================
cores_barras = {"Vendas": "#FFA500", "Receita": "#FF8C00", "Investimento": "#FF4500"}  # laranja degradê
cores_linhas = {"Vendas": "#1f77b4", "Receita": "#3b75c4", "Investimento": "#6fa8dc"}  # azul degradê

# Criar gráfico
fig = go.Figure()

for kpi in kpis:
    df_kpi = df_ano[df_ano['KPI'] == kpi]

    # Barras: metas (laranja)
    fig.add_trace(go.Bar(
        x=df_kpi['Mês'],
        y=df_kpi['Meta'],
        name=f"{kpi} - Meta",
        marker_color=cores_barras[kpi],
        opacity=0.6,
        showlegend=True  # legenda visível
    ))

    # Linhas: realizado (azul)
    fig.add_trace(go.Scatter(
        x=df_kpi['Mês'],
        y=df_kpi['Realizado'],
        name=f"{kpi} - Realizado",
        mode='lines+markers',  # sem +text
        marker=dict(color=cores_linhas[kpi], size=8),
        line=dict(width=3),
        hovertemplate=(
            f"<b>{kpi}</b><br>"
            "Mês: %{x}<br>"
            "Meta: R$ %{customdata[0]:,.2f}<br>"
            "Realizado: R$ %{y:,.2f}<extra></extra>"
        ),
        customdata=df_kpi[['Meta']],
        showlegend=True  # legenda visível
    ))

# Layout visual
fig.update_layout(
    title="Metas 2024 (Julho a Dezembro)",
    xaxis_title="Mês",
    yaxis_title="",
    barmode='group',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False, showticklabels=False),
    height=500
)

st.plotly_chart(fig, use_container_width=True)