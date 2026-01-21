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

# Protocols by attack type
st.subheader("Protocols by Attack")
protocols_by_attack = data['protocols_by_attack']


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

st.header("IP Address Statistics (only network data)")


st.write(f"Unique Source IPs : {data['ip_stats']['unique_sources']:,}")
st.write(f"Unique Destination IPs : {data['ip_stats']['unique_destinations']:,}")
st.write(f"Missing IPs : {data['ip_stats']['missing_ips']:,}")

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
        title='10 Most Active IP Addresses'
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.header("Sensor Statistics")

sensor_stats = data['sensor_stats']
if sensor_stats:
    for file_name, sensor_data in sensor_stats.items():
        with st.expander(f"{file_name}"):
            df_sensors = pd.DataFrame(
                sensor_data.items(),
                columns=['Label', 'Active Sensors']
            ).sort_values('Active Sensors', ascending=False)
            

            st.dataframe(df_sensors, use_container_width=True)

            fig = px.bar(
                df_sensors,
                x='Label',
                y='Active Sensors',
                title=f'Active Sensors by Label'
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sensor statistics available")

st.divider()

st.header("Missing Values")


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
        title='Missing Values in Network Data'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_missing_network, use_container_width=True)
else:
    st.success("No missing values in network datasets")


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
        title='Missing Values in Physical Data'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_missing_physical, use_container_width=True)
else:
    st.success("No missing values in physical datasets")

st.divider()