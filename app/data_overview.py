import streamlit as st

st.title("Data Overview")
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.summaries import get_all_data_for_streamlit

data = get_all_data_for_streamlit()


st.header("Statistics")


st.write(f"Total Rows : {data['global_stats']['total_rows']:,}")
st.write(f"Network Files : {data['global_stats']['network_files']}")
st.write(f"Physical Files : {data['global_stats']['physical_files']}")
st.write(f"Network Rows : {data['global_stats']['network_total_rows']:,}")
st.write(f"Physical Rows : {data['global_stats']['physical_total_rows']:,}")

st.divider()

# ============================================================================
# ATTACK VS NORMAL RATIO
# ============================================================================
st.header("Attack / Normal Traffic")


st.write(f"Normal Traffic : {data['attack_ratio']['normal']:,}")
st.write(f"Attack Traffic : {data['attack_ratio']['attacks']:,}")
st.write(f"Normal % : {data['attack_ratio']['normal_percentage']:.2f}%")
st.write(f"Attacks % : {data['attack_ratio']['attacks_percentage']:.2f}%")


fig = go.Figure(data=[go.Pie(
    labels=['Normal', 'Attacks'],
    values=[data['attack_ratio']['normal'], data['attack_ratio']['attacks']],
    hole=0.4,
    marker_colors=['lightgreen', 'darksalmon']
)])
fig.update_layout(
    title="Data Distribution",
    height=300
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Label Distribution")

labels_combined = data['label_distribution']['combined']
df_labels_combined = pd.DataFrame(
    labels_combined.items(),
    columns=['Label', 'Count']
).sort_values('Count', ascending=False)

fig = px.bar(
    df_labels_combined,
    x='Label',
    y='Count',
    title='All Label Distribution'
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_labels_combined, use_container_width=True)

labels_network = data['label_distribution']['network']
df_labels_network = pd.DataFrame(
    labels_network.items(),
    columns=['Label', 'Count']
).sort_values('Count', ascending=False)

fig = px.bar(
    df_labels_network,
    x='Label',
    y='Count',
    title='Network Label Distribution'
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_labels_network, use_container_width=True)

labels_physical = data['label_distribution']['physical']
df_labels_physical = pd.DataFrame(
    labels_physical.items(),
    columns=['Label', 'Count']
).sort_values('Count', ascending=False)

fig = px.bar(
    df_labels_physical,
    x='Label',
    y='Count',
    title='Physical Label Distribution'
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_labels_physical, use_container_width=True)

st.divider()

st.header("Protocol Analysis")

protocols = data['protocols']
df_protocols = pd.DataFrame(
    protocols.items(),
    columns=['Protocol', 'Count']
).sort_values('Count', ascending=False)

fig = px.bar(
    df_protocols,
    x='Protocol',
    y='Count',
    title='Protocols counts'
)
st.plotly_chart(fig, use_container_width=True)


st.subheader("Top 10 Protocols")
for protocol, count in data['top_protocols']:
    st.write(f"protocol  : {count:,}")

# Protocols by attack type
st.subheader("Protocols by Attack Type")
protocols_by_attack = data['protocols_by_attack']

if protocols_by_attack:
    selected_attack = st.selectbox(
        "Select attack type",
        options=list(protocols_by_attack.keys())
    )
    
    if selected_attack and protocols_by_attack[selected_attack]:
        df_attack_protocols = pd.DataFrame(
            protocols_by_attack[selected_attack].items(),
            columns=['Protocol', 'Count']
        ).sort_values('Count', ascending=False)
        
        fig = px.pie(
            df_attack_protocols,
            names='Protocol',
            values='Count',
            title=f'Protocol Distribution for {selected_attack}'
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================================
# IP STATISTICS (Network only)
# ============================================================================
st.header("🌐 IP Address Statistics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Unique Source IPs", f"{data['ip_stats']['unique_sources']:,}")
with col2:
    st.metric("Unique Destination IPs", f"{data['ip_stats']['unique_destinations']:,}")
with col3:
    st.metric("Missing IPs", f"{data['ip_stats']['missing_ips']:,}")

st.subheader("Top 10 Most Active IPs")
if data['top_ips']:
    df_top_ips = pd.DataFrame(
        data['top_ips'],
        columns=['IP Address', 'Count']
    )
    
    fig = px.bar(
        df_top_ips,
        x='IP Address',
        y='Count',
        title='Most Active IP Addresses',
        color='Count',
        color_continuous_scale='Oranges'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================================
# TEMPORAL ANALYSIS
# ============================================================================
st.header("⏰ Temporal Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Network Time Ranges")
    if data['time_ranges']['network']:
        for i, time_range in enumerate(data['time_ranges']['network'], 1):
            st.text(f"{i}. {time_range}")
    else:
        st.info("No time range data available")

with col2:
    st.subheader("Physical Time Ranges")
    if data['time_ranges']['physical']:
        for i, time_range in enumerate(data['time_ranges']['physical'], 1):
            st.text(f"{i}. {time_range}")
    else:
        st.info("No time range data available")

# Hourly distribution
st.subheader("Hourly Event Distribution")

tab1, tab2 = st.tabs(["Network", "Physical"])

with tab1:
    hourly_network = data['hourly_dist']['network']
    if hourly_network:
        df_hourly_network = pd.DataFrame(
            [(hour, count) for hour, count in sorted(hourly_network.items())],
            columns=['Hour', 'Count']
        )
        
        fig = px.line(
            df_hourly_network,
            x='Hour',
            y='Count',
            title='Network Traffic by Hour',
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hourly data available")

with tab2:
    hourly_physical = data['hourly_dist']['physical']
    if hourly_physical:
        df_hourly_physical = pd.DataFrame(
            [(hour, count) for hour, count in sorted(hourly_physical.items())],
            columns=['Hour', 'Count']
        )
        
        fig = px.line(
            df_hourly_physical,
            x='Hour',
            y='Count',
            title='Physical Sensor Events by Hour',
            markers=True
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hourly data available")

st.divider()

# ============================================================================
# SENSOR STATISTICS (Physical only)
# ============================================================================
st.header("📡 Sensor Statistics")

sensor_stats = data['sensor_stats']
if sensor_stats:
    for file_name, sensor_data in sensor_stats.items():
        with st.expander(f"📄 {file_name}"):
            df_sensors = pd.DataFrame(
                sensor_data.items(),
                columns=['Label', 'Active Sensors']
            ).sort_values('Active Sensors', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(df_sensors, use_container_width=True)
            with col2:
                fig = px.bar(
                    df_sensors,
                    x='Label',
                    y='Active Sensors',
                    title=f'Active Sensors by Label',
                    color='Active Sensors'
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sensor statistics available")

st.divider()

# ============================================================================
# MISSING VALUES
# ============================================================================
st.header("❓ Missing Values Analysis")

tab1, tab2 = st.tabs(["Network", "Physical"])

with tab1:
    missing_network = data['missing_values']['network']
    if missing_network:
        df_missing_network = pd.DataFrame(
            missing_network.items(),
            columns=['Column', 'Missing Count']
        ).sort_values('Missing Count', ascending=False)
        
        fig = px.bar(
            df_missing_network,
            x='Column',
            y='Missing Count',
            title='Missing Values in Network Data',
            color='Missing Count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_missing_network, use_container_width=True)
    else:
        st.success("✅ No missing values in network datasets!")

with tab2:
    missing_physical = data['missing_values']['physical']
    if missing_physical:
        df_missing_physical = pd.DataFrame(
            missing_physical.items(),
            columns=['Column', 'Missing Count']
        ).sort_values('Missing Count', ascending=False)
        
        fig = px.bar(
            df_missing_physical,
            x='Column',
            y='Missing Count',
            title='Missing Values in Physical Data',
            color='Missing Count',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_missing_physical, use_container_width=True)
    else:
        st.success("✅ No missing values in physical datasets!")

st.divider()

# ============================================================================
# DATASET SUMMARY TABLE
# ============================================================================
st.header("📋 Dataset Summary Table")

df_summary = pd.DataFrame(data['summary_table'])

# Add styling
st.dataframe(
    df_summary.style.background_gradient(subset=['Rows'], cmap='YlOrRd'),
    use_container_width=True,
    height=400
)

# Download button
csv = df_summary.to_csv(index=False)
st.download_button(
    label="📥 Download Summary as CSV",
    data=csv,
    file_name="dataset_summary.csv",
    mime="text/csv"
)