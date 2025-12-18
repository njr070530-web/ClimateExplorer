import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import numpy as np
from datetime import datetime, timedelta
import os
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk-KcKAu5U9JBEbbq0wReeTeFVm60lMqvrmYqjDYQgQ4MHF6Jl2'
os.environ['OPENAI_API_BASE'] = 'https://api3.wlai.vip/v1'

def load_and_inspect_data():
    try:
        print(f"当前工作目录是: {os.getcwd()}")
        print(f"该目录下包含的文件: {os.listdir('.')}")
        # CO2
        print("Reading CO2 data...")
        co2_data = pd.read_csv('./data/co2_1880_2023.csv')
        # city temperature
        print("Reading city temperature data...")
        city_temp_data = pd.read_csv('./data/GlobalLandTemperaturesByCity.csv')
        # country temperature
        print("Reading country temperature data...")
        country_temp_data = pd.read_csv('./data/GlobalLandTemperaturesByCountry.csv')
        return co2_data, city_temp_data, country_temp_data

    except Exception as e:
        sys.exit(f"Error when reading datas: {e}")

def preprocess_data(co2_data, city_temp_data, country_temp_data):
    # tackle co2 data
    co2_columns = co2_data.columns.tolist()
    if len(co2_columns) >= 2:
        co2_data_processed = co2_data.rename(columns={
            co2_columns[0]: 'Year',
            co2_columns[1]: 'ppm'
        })
    else:
        co2_data_processed = co2_data.copy()
        if 'Year' not in co2_data_processed.columns:
            co2_data_processed['Year'] = range(1880, 1880 + len(co2_data_processed))
        if 'ppm' not in co2_data_processed.columns:
            co2_data_processed['ppm'] = 280 + (co2_data_processed.index * 0.5)  # 模拟数据
    # tackle temperature
    city_temp_processed = city_temp_data.copy()
    country_temp_processed = country_temp_data.copy()

    for df in [city_temp_processed, country_temp_processed]:
        # dates
        date_col = None
        for col in ['dt', 'Date', 'date', 'time']:
            if col in df.columns:
                date_col = col
                break

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df['Year'] = df[date_col].dt.year
            df['Month'] = df[date_col].dt.month
        else:
            # 如果没有日期列，创建模拟数据
            df['Year'] = np.random.randint(1950, 2014, len(df))
            df['Month'] = np.random.randint(1, 13, len(df))

    return co2_data_processed, city_temp_processed, country_temp_processed

print("Loading datas...")
co2_data, city_temp_data, country_temp_data = load_and_inspect_data()

co2_data, city_temp_data, country_temp_data = preprocess_data(
    co2_data, city_temp_data, country_temp_data
)

def generate_predictions(data, data_type='temperature', years_to_predict=10):

    try:
        if data_type == 'co2':
            latest_year = data['Year'].max()
            future_years = range(latest_year + 1, latest_year + years_to_predict + 1)

            #simple linear induction
            recent_data = data[data['Year'] >= latest_year - 10]
            recent_avg = recent_data['ppm'].mean()
            trend = 2.0  # 假设每年上升2 ppm
            predictions = [recent_avg + trend * (year - latest_year) for year in future_years]

            return pd.DataFrame({
                'Year': list(future_years),
                'Predicted_ppm': predictions,
                'Type': 'Prediction'
            })

        else:  # temperature
            if 'Country' in data.columns:
                yearly_data = data.groupby('Year')['AverageTemperature'].mean().reset_index()
            else:
                yearly_data = data.copy()

            latest_year = yearly_data['Year'].max()
            future_years = range(latest_year + 1, latest_year + years_to_predict + 1)

            # simple linear induction
            recent_data = yearly_data[yearly_data['Year'] >= latest_year - 10]
            recent_avg = recent_data['AverageTemperature'].mean()
            trend = 0.02
            predictions = [recent_avg + trend * (year - latest_year) for year in future_years]

            return pd.DataFrame({
                'Year': list(future_years),
                'Predicted': predictions,
                'Type': 'Prediction'
            })

    except Exception as e:
        print(f"Error when generating prediction: {e}")
        return pd.DataFrame()

# generate prediction
print("Generating predictions...")
co2_predictions = generate_predictions(co2_data, 'co2')
country_predictions = generate_predictions(country_temp_data, 'temperature')

print("Succeeded generating predictions...")
print("CO2 prediction:", co2_predictions.shape)
print("Temperature prediction:", country_predictions.shape)


# 找到原有的 generate_llm_insight 函数，替换成这个：
def generate_llm_insight(selected_country, selected_year, metric_type):
    # 获取配置好的环境变量
    api_key = os.environ.get('OPENAI_API_KEY')
    base_url = os.environ.get('OPENAI_API_BASE')
    
    # print一下看看有没有读到 (调试用)
    print(f"正在连接服务器: {base_url} ...")

    # 初始化客户端 (关键修改：加入了 base_url)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url  # <--- 这就是老师说的 API_BASE，必须加！
    )
    
    print(f"正在呼叫 GPT-5.1 分析 {selected_country}...")

    # 2. 构造一个动态的 Prompt，把网页上的数据喂给 AI
    prompt = f"""
    你是一个专业的气候学家。
    用户正在查看全球气候仪表盘。
    当前选择的国家是：{selected_country}
    当前关注的时间节点是：{selected_year}年
    当前关注的指标是：{'CO2浓度' if metric_type == 'co2' else '平均气温'}
    
    请根据这些信息，用简练、专业的语言（英文和中文），生成一段约100字的分析洞察。
    需要包含：该国/该指标的历史趋势评价，以及对未来的简短科学预测或警示。
    不要说“根据数据”，直接给出结论。
    """

    try:
        # 3. 调用 API
        response = client.chat.completions.create(
            model="gpt-5.1",  # <--- 使用你的最强模型
            messages=[
                {"role": "system", "content": "You are a helpful climate data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 分析暂时不可用: {e}"

    # Add regional insight if country is specified
    if selected_country:
        country_insight = regional_patterns.get(selected_country,
                                              f"{selected_country} exhibits climate trends influenced by both global warming patterns and regional geographic characteristics.")
        base_insights.append(country_insight)

    # Add temporal context
    if selected_year:
        if selected_year >= 2000:
            base_insights.append(f"Since {selected_year}, warming acceleration has become more pronounced, with multiple climate records broken and polar amplification intensifying.")
        elif selected_year >= 1950:
            base_insights.append(f"The period since {selected_year} has seen accelerated warming coinciding with rapid industrialization and increased fossil fuel consumption globally.")

    # Add metric-specific insights
    if metric_type == 'co2':
        base_insights.append("Current CO₂ levels exceed pre-industrial concentrations by over 50%, reaching the highest levels in at least 800,000 years, with direct implications for climate sensitivity and future warming commitments.")
    elif metric_type == 'temperature':
        base_insights.append("The past five decades represent the warmest period in the Northern Hemisphere in nearly 2000 years, with warming trends showing strong spatial heterogeneity across climate regions.")

    # Policy and research implications
    policy_insights = [
        "Policy Recommendation: Enhance emission reduction cooperation among high-latitude nations to address accelerated polar warming and its global impacts.",
        "Adaptation Strategy: Develop climate zone-specific governance systems to improve infrastructure resilience across different vulnerability profiles.",
        "Research Priority: Integrate satellite data, paleoclimate proxies, and climate models to improve attribution analysis and address data coverage limitations in early records."
    ]

    # Combine base insights with policy recommendations
    final_insights = base_insights[:2] + policy_insights[:1]  # 2 technical insights + 1 policy insight

    return " ".join(final_insights)


# 创建Dash应用
app = dash.Dash(__name__)

# 应用布局
app.layout = html.Div([
    html.H1("🌍 Global temperature interactive dashboard",
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

    # 控制面板
    html.Div([
        html.Div([
            html.Label("📍 country:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country}
                         for country in sorted(country_temp_data['Country'].unique())],  # 限制数量避免性能问题
                value='United States' if 'United States' in country_temp_data['Country'].values else
                sorted(country_temp_data['Country'].unique())[0],
                style={'width': '100%'}
            ),
        ], style={'width': '30%', 'display': 'inline-block', 'padding': '10px'}),

        html.Div([
            html.Label("📅 time range:", style={'fontWeight': 'bold'}),
            dcc.RangeSlider(
                id='year-slider',
                min=int(co2_data['Year'].min()),
                max=int(co2_data['Year'].max()),
                step=10,
                marks={year: str(year) for year in
                       range(int(co2_data['Year'].min()), int(co2_data['Year'].max()) + 1, 50)},
                value=[1990, 2020]
            ),
        ], style={'width': '65%', 'display': 'inline-block', 'padding': '10px'}),
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px'}),

    html.Div([
        html.H3("💡 climate insights", style={'color': '#34495e'}),
        html.Div(id='llm-insight', style={
            'padding': '15px',
            'backgroundColor': '#e8f4fd',
            'borderRadius': '8px',
            'borderLeft': '5px solid #3498db',
            'marginBottom': '20px',
            'fontSize': '16px',
            'lineHeight': '1.6'
        })
    ]),

    dcc.Tabs([
        dcc.Tab(label='🌡️ temperature trend', children=[
            html.Div([
                dcc.Graph(id='temperature-trend'),
                dcc.Graph(id='country-temperature-trend')
            ], style={'padding': '20px'})
        ]),

        # CO2趋势标签页
        dcc.Tab(label='💨 co2 density', children=[
            html.Div([
                dcc.Graph(id='co2-trend'),
                html.Div([
                    dcc.Graph(id='co2-temperature-correlation', style={'width': '48%', 'display': 'inline-block'}),
                    dcc.Graph(id='monthly-co2-trend', style={'width': '48%', 'display': 'inline-block'})
                ])
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='🗺️ geographical distribution', children=[
            html.Div([
                dcc.Graph(id='world-heatmap'),
                html.Div([
                    html.Label("📅 year:", style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Slider(
                        id='heatmap-year-slider',
                        min=int(country_temp_data['Year'].min()),
                        max=int(country_temp_data['Year'].max()),
                        step=10,
                        value=2000,
                        marks={year: str(year) for year in
                               range(int(country_temp_data['Year'].min()), int(country_temp_data['Year'].max()) + 1,
                                     50)}
                    )
                ], style={'padding': '20px'})
            ])
        ])
    ])
])


# 回调函数
@app.callback(
    [Output('temperature-trend', 'figure'),
     Output('country-temperature-trend', 'figure'),
     Output('co2-trend', 'figure'),
     Output('co2-temperature-correlation', 'figure'),
     Output('monthly-co2-trend', 'figure'),
     Output('world-heatmap', 'figure'),
     Output('llm-insight', 'children')],
    [Input('country-dropdown', 'value'),
     Input('year-slider', 'value'),
     Input('heatmap-year-slider', 'value')]
)
def update_dashboard(selected_country, year_range, heatmap_year):
    start_year, end_year = year_range

    try:
        # 1. 全球温度趋势图
        if 'AverageTemperature' in country_temp_data.columns:
            global_temp_trend = country_temp_data.groupby('Year')['AverageTemperature'].mean().reset_index()
        else:
            # 如果没有温度列，创建模拟数据
            global_temp_trend = pd.DataFrame({
                'Year': range(int(country_temp_data['Year'].min()), int(country_temp_data['Year'].max()) + 1),
                'AverageTemperature': [10 + 0.01 * (year - 1950) for year in range(int(country_temp_data['Year'].min()),
                                                                                   int(country_temp_data[
                                                                                           'Year'].max()) + 1)]
            })

        global_temp_filtered = global_temp_trend[
            (global_temp_trend['Year'] >= start_year) & (global_temp_trend['Year'] <= end_year)
            ]

        fig_global_temp = go.Figure()
        fig_global_temp.add_trace(go.Scatter(
            x=global_temp_filtered['Year'],
            y=global_temp_filtered['AverageTemperature'],
            mode='lines',
            name='historical data',
            line=dict(color='blue', width=2)
        ))

        # 添加预测
        if not country_predictions.empty:
            prediction_period = country_predictions[
                (country_predictions['Year'] > end_year) &
                (country_predictions['Year'] <= end_year + 10)
                ]
            if not prediction_period.empty:
                fig_global_temp.add_trace(go.Scatter(
                    x=prediction_period['Year'],
                    y=prediction_period['Predicted'],
                    mode='lines',
                    name='prediction',
                    line=dict(color='red', width=2, dash='dash')
                ))

        fig_global_temp.update_layout(
            title='global average temperature（including prediction）',
            xaxis_title='year',
            yaxis_title='temperature(°C)',
            hovermode='x unified'
        )

        # 2. 选定国家温度趋势
        if 'Country' in country_temp_data.columns and 'AverageTemperature' in country_temp_data.columns:
            country_data = country_temp_data[
                (country_temp_data['Country'] == selected_country) &
                (country_temp_data['Year'] >= start_year) &
                (country_temp_data['Year'] <= end_year)
                ].groupby('Year')['AverageTemperature'].mean().reset_index()
        else:
            country_data = pd.DataFrame()  # 空DataFrame

        fig_country_temp = go.Figure()
        if not country_data.empty:
            fig_country_temp.add_trace(go.Scatter(
                x=country_data['Year'],
                y=country_data['AverageTemperature'],
                mode='lines+markers',
                name=f'{selected_country}温度',
                line=dict(color='green', width=2)
            ))
        else:
            # 如果没有数据，显示提示
            fig_country_temp.add_annotation(
                text=f"We can't find data from {selected_country}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        fig_country_temp.update_layout(
            title=f'{selected_country} temperature trend',
            xaxis_title='year',
            yaxis_title='temperature(°C)'
        )

        # 3. CO2趋势图
        co2_filtered = co2_data[
            (co2_data['Year'] >= start_year) & (co2_data['Year'] <= end_year)
            ]

        fig_co2 = go.Figure()
        if not co2_filtered.empty:
            fig_co2.add_trace(go.Scatter(
                x=co2_filtered['Year'],
                y=co2_filtered['ppm'],
                mode='lines',
                name='CO2 historical data',
                line=dict(color='orange', width=2)
            ))

        # add co2 prediction
        if not co2_predictions.empty:
            co2_prediction_period = co2_predictions[
                (co2_predictions['Year'] > end_year) &
                (co2_predictions['Year'] <= end_year + 10)
                ]
            if not co2_prediction_period.empty:
                fig_co2.add_trace(go.Scatter(
                    x=co2_prediction_period['Year'],
                    y=co2_prediction_period['Predicted_ppm'],
                    mode='lines',
                    name='CO2 prediction',
                    line=dict(color='red', width=2, dash='dash')
                ))

        fig_co2.update_layout(
            title='CO2 density trend（including prediction）',
            xaxis_title='year',
            yaxis_title='CO2 density (ppm)'
        )

        # relavance
        merged_data = pd.merge(
            global_temp_trend, co2_data, on='Year', how='inner'
        )
        merged_data = merged_data[
            (merged_data['Year'] >= start_year) & (merged_data['Year'] <= end_year)
            ]

        if not merged_data.empty and len(merged_data) > 5:
            fig_correlation = px.scatter(
                merged_data, x='ppm', y='AverageTemperature',
                trendline='lowess',
                title='relevance between co2 density and temperature',
                labels={'ppm': 'CO2 density (ppm)', 'AverageTemperature': 'global average temperature (°C)'}
            )
        else:
            fig_correlation = go.Figure()
            fig_correlation.add_annotation(
                text="No enough data for relevance evaluation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig_correlation.update_layout(title='relevance between co2 density and temperatur')

        # 5. co2 trend (monthly)
        months = ['1月', '2月', '3月', '4月', '5月', '6月',
                  '7月', '8月', '9月', '10月', '11月', '12月']
        monthly_co2 = pd.DataFrame({
            'Month': months,
            'CO2_ppm': [415 + 10 * np.sin(2 * np.pi * i / 12) for i in range(12)]
        })

        fig_monthly_co2 = px.line(
            monthly_co2, x='Month', y='CO2_ppm',
            title='CO2 density change in season',
            markers=True
        )

        if 'Country' in country_temp_data.columns and 'AverageTemperature' in country_temp_data.columns:
            heatmap_data = country_temp_data[
                (country_temp_data['Year'] == heatmap_year)
            ].groupby('Country')['AverageTemperature'].mean().reset_index()
        else:
            heatmap_data = pd.DataFrame()

        if not heatmap_data.empty:
            fig_world_heatmap = px.choropleth(
                heatmap_data,
                locations='Country',
                locationmode='country names',
                color='AverageTemperature',
                hover_name='Country',
                color_continuous_scale='RdBu_r',
                title=f'{heatmap_year} global heatmap',
            )
        else:
            fig_world_heatmap = go.Figure()
            fig_world_heatmap.add_annotation(
                text=f"No temperature data in {heatmap_year}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig_world_heatmap.update_layout(title=f'{heatmap_year} global temperatur')

        # 7. LLM叙事洞察
        llm_insight = generate_llm_insight(selected_country, end_year, 'temperature')

        return (fig_global_temp, fig_country_temp, fig_co2, fig_correlation,
                fig_monthly_co2, fig_world_heatmap, llm_insight)

    except Exception as e:
        print(f"Error when updating dashboard: {e}")
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return (error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, f"Error: {str(e)}")


# 运行应用
if __name__ == '__main__':
    print("=" * 50)
    print("Launch the climate change dashboard...")
    print(f"Data range: {co2_data['Year'].min()} - {co2_data['Year'].max()}")
    print(
        f"Available Countries number: {len(country_temp_data['Country'].unique()) if 'Country' in country_temp_data.columns else 'N/A'}")
    print("Visit: http://localhost:8050")
    print("=" * 50)

    # 使用新的运行方法
    app.run(debug=True, port=8050)