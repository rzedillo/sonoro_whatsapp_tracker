"""
Streamlit Frontend for WhatsApp Task Tracker
Enhanced Framework V3.1 Implementation
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="WhatsApp Task Tracker",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend API configuration
BACKEND_URL = "http://backend:8000"  # Docker service name
if "backend_url" not in st.session_state:
    st.session_state.backend_url = BACKEND_URL


class APIClient:
    """API client for backend communication"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """GET request to backend"""
        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """POST request to backend"""
        try:
            response = requests.post(f"{self.base_url}{endpoint}", json=data)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def put(self, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """PUT request to backend"""
        try:
            response = requests.put(f"{self.base_url}{endpoint}", json=data)
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient(st.session_state.backend_url)


api = get_api_client()


def main():
    """Main application function"""
    
    # Sidebar navigation
    st.sidebar.title("🤖 WhatsApp Task Tracker")
    st.sidebar.markdown("---")
    
    # Navigation menu
    pages = {
        "📊 Dashboard": show_dashboard,
        "📋 Tasks": show_tasks,
        "📱 WhatsApp": show_whatsapp,
        "📈 Analytics": show_analytics,
        "🔔 Notifications": show_notifications,
        "⚙️ Agents": show_agents,
        "🛠️ Settings": show_settings
    }
    
    selected_page = st.sidebar.selectbox("Navigate to", list(pages.keys()))
    
    # System status in sidebar
    show_system_status()
    
    # Main content
    pages[selected_page]()


def show_system_status():
    """Show system status in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 System Status")
    
    # Get system health
    health = api.get("/health")
    
    if health.get("status") == "healthy":
        st.sidebar.success("✅ System Healthy")
    else:
        st.sidebar.error("❌ System Issues")
    
    # WhatsApp status
    whatsapp_status = api.get("/api/v1/whatsapp/health")
    if whatsapp_status.get("healthy"):
        st.sidebar.success("📱 WhatsApp Connected")
    else:
        st.sidebar.warning("📱 WhatsApp Disconnected")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Status"):
        st.experimental_rerun()


def show_dashboard():
    """Main dashboard page"""
    st.title("📊 Dashboard")
    st.markdown("Welcome to your WhatsApp Task Tracker dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get task summary
    summary = api.get("/api/v1/tasks/stats/summary")
    
    if summary.get("success"):
        data = summary.get("data", {})
        
        with col1:
            st.metric(
                "Total Tasks",
                data.get("total_tasks", 0),
                delta=None
            )
        
        with col2:
            st.metric(
                "Completed",
                data.get("completed_tasks", 0),
                delta=f"{data.get('completion_rate', 0):.1%}"
            )
        
        with col3:
            st.metric(
                "Pending",
                data.get("pending_tasks", 0),
                delta=None
            )
        
        with col4:
            st.metric(
                "Groups",
                len(data.get("user_breakdown", {})),
                delta=None
            )
    
    # Recent activity
    st.subheader("📝 Recent Activity")
    
    # Get recent tasks
    recent_tasks = api.get("/api/v1/tasks/", params={"limit": 10})
    
    if recent_tasks.get("success"):
        tasks = recent_tasks.get("data", {}).get("tasks", [])
        
        if tasks:
            # Convert to DataFrame for display
            df = pd.DataFrame(tasks)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            # Display table
            st.dataframe(
                df[["descripcion", "responsable", "prioridad", "estado", "timestamp"]],
                use_container_width=True
            )
        else:
            st.info("No recent tasks found")
    else:
        st.error("Failed to load recent tasks")
    
    # Task creation chart
    st.subheader("📈 Task Creation Trend")
    
    if recent_tasks.get("success") and tasks:
        df = pd.DataFrame(tasks)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        
        # Count tasks per day
        daily_counts = df.groupby("date").size().reset_index(name="count")
        
        fig = px.line(
            daily_counts,
            x="date",
            y="count",
            title="Tasks Created Per Day",
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_tasks():
    """Tasks management page"""
    st.title("📋 Task Management")
    
    # Task filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "pendiente", "en_progreso", "completado", "cancelado"]
        )
    
    with col2:
        priority_filter = st.selectbox(
            "Priority",
            ["All", "baja", "media", "alta", "urgente"]
        )
    
    with col3:
        assigned_filter = st.text_input("Assigned to")
    
    # Build filters
    filters = {}
    if status_filter != "All":
        filters["status"] = status_filter
    if priority_filter != "All":
        filters["priority"] = priority_filter
    if assigned_filter:
        filters["assigned_to"] = assigned_filter
    
    # Get tasks
    tasks_response = api.get("/api/v1/tasks/", params=filters)
    
    if tasks_response.get("success"):
        tasks = tasks_response.get("data", {}).get("tasks", [])
        
        if tasks:
            # Create tabs for different views
            tab1, tab2, tab3 = st.tabs(["📋 List View", "📊 Analytics", "➕ Create Task"])
            
            with tab1:
                show_tasks_list(tasks)
            
            with tab2:
                show_tasks_analytics(tasks)
            
            with tab3:
                show_create_task()
        else:
            st.info("No tasks found with current filters")
            
            # Show create task form
            st.subheader("➕ Create New Task")
            show_create_task()
    else:
        st.error(f"Failed to load tasks: {tasks_response.get('error', 'Unknown error')}")


def show_tasks_list(tasks: List[Dict]):
    """Display tasks in list format"""
    
    for task in tasks:
        with st.expander(f"🔖 {task['descripcion'][:50]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Description:** {task['descripcion']}")
                st.write(f"**Assigned to:** {task.get('responsable', 'Unassigned')}")
                st.write(f"**Group:** {task.get('grupo_nombre', 'N/A')}")
                st.write(f"**Created:** {task.get('timestamp', 'N/A')}")
                
                if task.get("fecha_limite"):
                    st.write(f"**Due Date:** {task['fecha_limite']}")
            
            with col2:
                # Status badge
                status_color = {
                    "pendiente": "🟡",
                    "en_progreso": "🔵",
                    "completado": "🟢",
                    "cancelado": "🔴"
                }
                st.write(f"**Status:** {status_color.get(task['estado'], '⚪')} {task['estado']}")
                
                # Priority badge
                priority_color = {
                    "baja": "🟢",
                    "media": "🟡",
                    "alta": "🟠",
                    "urgente": "🔴"
                }
                st.write(f"**Priority:** {priority_color.get(task['prioridad'], '⚪')} {task['prioridad']}")
                
                # Action buttons
                if task['estado'] == 'pendiente':
                    if st.button(f"✅ Complete", key=f"complete_{task['id']}"):
                        complete_task(task['id'])
                
                if st.button(f"✏️ Edit", key=f"edit_{task['id']}"):
                    show_edit_task(task)


def show_tasks_analytics(tasks: List[Dict]):
    """Show task analytics"""
    if not tasks:
        st.info("No tasks to analyze")
        return
    
    df = pd.DataFrame(tasks)
    
    # Status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        status_counts = df['estado'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Task Status Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        priority_counts = df['prioridad'].value_counts()
        fig = px.bar(
            x=priority_counts.index,
            y=priority_counts.values,
            title="Task Priority Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Assignee analysis
    if 'responsable' in df.columns:
        st.subheader("👥 Assignee Analysis")
        assignee_stats = df.groupby('responsable').agg({
            'id': 'count',
            'estado': lambda x: (x == 'completado').sum()
        }).rename(columns={'id': 'total_tasks', 'estado': 'completed_tasks'})
        
        assignee_stats['completion_rate'] = assignee_stats['completed_tasks'] / assignee_stats['total_tasks']
        
        st.dataframe(assignee_stats, use_container_width=True)


def show_create_task():
    """Show create task form"""
    with st.form("create_task"):
        col1, col2 = st.columns(2)
        
        with col1:
            descripcion = st.text_area("Task Description", height=100)
            responsable = st.text_input("Assigned To")
            grupo_nombre = st.text_input("Group Name")
        
        with col2:
            prioridad = st.selectbox("Priority", ["baja", "media", "alta", "urgente"])
            fecha_limite = st.date_input("Due Date", value=None)
            autor_mensaje = st.text_input("Created By", value="manual")
        
        submitted = st.form_submit_button("🚀 Create Task")
        
        if submitted and descripcion:
            task_data = {
                "descripcion": descripcion,
                "responsable": responsable if responsable else None,
                "prioridad": prioridad,
                "grupo_nombre": grupo_nombre if grupo_nombre else None,
                "fecha_limite": fecha_limite.isoformat() if fecha_limite else None,
                "autor_mensaje": autor_mensaje
            }
            
            result = api.post("/api/v1/tasks/", task_data)
            
            if result.get("success"):
                st.success("✅ Task created successfully!")
                st.experimental_rerun()
            else:
                st.error(f"❌ Failed to create task: {result.get('error', 'Unknown error')}")


def complete_task(task_id: int):
    """Mark task as completed"""
    result = api.post(f"/api/v1/tasks/{task_id}/complete")
    
    if result.get("success"):
        st.success("✅ Task completed!")
        st.experimental_rerun()
    else:
        st.error(f"❌ Failed to complete task: {result.get('error', 'Unknown error')}")


def show_edit_task(task: Dict):
    """Show edit task form"""
    st.subheader(f"✏️ Edit Task: {task['descripcion'][:30]}...")
    
    # This would open a modal or form to edit the task
    # For now, just show the task details
    st.json(task)


def show_whatsapp():
    """WhatsApp management page"""
    st.title("📱 WhatsApp Connection")
    
    # Get WhatsApp status
    status = api.get("/api/v1/whatsapp/status")
    
    if status.get("success"):
        is_authenticated = status.get("authenticated", False)
        
        if is_authenticated:
            st.success("✅ WhatsApp is connected and authenticated")
            
            # Show session info
            session_info = status
            st.write(f"**Phone Number:** {session_info.get('phone_number', 'Unknown')}")
            st.write(f"**Session ID:** {session_info.get('session_id', 'Unknown')}")
            st.write(f"**Monitored Groups:** {len(session_info.get('monitored_groups', []))}")
            
            # Reconnect button
            if st.button("🔄 Reconnect"):
                with st.spinner("Reconnecting..."):
                    result = api.post("/api/v1/whatsapp/reconnect")
                    if result.get("success"):
                        st.success("✅ Reconnected successfully")
                        st.experimental_rerun()
                    else:
                        st.error(f"❌ Reconnection failed: {result.get('error')}")
        
        else:
            st.warning("⚠️ WhatsApp not authenticated")
            
            # Get QR code
            qr_response = api.get("/api/v1/whatsapp/qr-code")
            
            if qr_response.get("success"):
                qr_code = qr_response.get("data", {}).get("qr_code")
                
                if qr_code:
                    st.subheader("📱 Scan QR Code")
                    st.markdown("Scan this QR code with your WhatsApp mobile app:")
                    
                    # Display QR code (base64 image)
                    import base64
                    from io import BytesIO
                    
                    try:
                        # Decode base64 QR code
                        qr_data = base64.b64decode(qr_code)
                        st.image(qr_data, caption="WhatsApp QR Code", width=300)
                        
                        # Auto-refresh
                        if st.button("🔄 Refresh QR Code"):
                            st.experimental_rerun()
                        
                        # Auto refresh every 10 seconds
                        time.sleep(10)
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Failed to display QR code: {str(e)}")
                else:
                    st.info("QR code not available. Please check WhatsApp service.")
            else:
                st.error("Failed to get QR code")
    
    else:
        st.error(f"Failed to get WhatsApp status: {status.get('error', 'Unknown error')}")
    
    # Recent conversations
    st.subheader("💬 Recent Conversations")
    
    conversations = api.get("/api/v1/whatsapp/conversations", params={"limit": 20})
    
    if conversations.get("success"):
        conv_data = conversations.get("data", {}).get("conversations", [])
        
        if conv_data:
            df = pd.DataFrame(conv_data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            st.dataframe(
                df[["mensaje", "autor", "grupo_nome", "timestamp"]],
                use_container_width=True
            )
        else:
            st.info("No recent conversations found")
    else:
        st.error("Failed to load conversations")


def show_analytics():
    """Analytics page"""
    st.title("📈 Analytics & Insights")
    
    # Time period selector
    period = st.selectbox("Analysis Period", [7, 14, 30, 60, 90], index=2)
    
    # Get productivity analytics
    analytics = api.get("/api/v1/analytics/productivity", params={"days": period})
    
    if analytics.get("success"):
        data = analytics
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Completion Rate", f"{data.get('completion_rate', 0):.1%}")
        
        with col2:
            st.metric("Total Tasks", data.get("total_tasks", 0))
        
        with col3:
            st.metric("Completed", data.get("completed_tasks", 0))
        
        with col4:
            st.metric("Pending", data.get("pending_tasks", 0))
        
        # User performance
        st.subheader("👥 User Performance")
        
        user_breakdown = data.get("user_breakdown", {})
        if user_breakdown:
            user_data = []
            for user, stats in user_breakdown.items():
                user_data.append({
                    "User": user,
                    "Total Tasks": stats["total"],
                    "Completed": stats["completed"],
                    "Completion Rate": stats["completed"] / stats["total"] if stats["total"] > 0 else 0
                })
            
            df = pd.DataFrame(user_data)
            
            # Bar chart
            fig = px.bar(
                df,
                x="User",
                y=["Total Tasks", "Completed"],
                title="Tasks by User",
                barmode="group"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Completion rate chart
            fig2 = px.bar(
                df,
                x="User",
                y="Completion Rate",
                title="Completion Rate by User"
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.error("Failed to load analytics data")


def show_notifications():
    """Notifications page"""
    st.title("🔔 Notifications")
    
    # Get notifications
    notifications = api.get("/api/v1/notifications/")
    
    if notifications.get("success"):
        notif_data = notifications.get("data", {}).get("notifications", [])
        
        if notif_data:
            for notif in notif_data:
                with st.expander(f"{notif.get('title', 'Notification')} - {notif.get('timestamp', '')}"):
                    st.write(f"**Type:** {notif.get('type', 'Unknown')}")
                    st.write(f"**Message:** {notif.get('message', '')}")
                    st.write(f"**Priority:** {notif.get('priority', 'medium')}")
                    
                    if notif.get("details"):
                        st.json(notif["details"])
        else:
            st.info("No notifications available")
    
    else:
        st.error("Failed to load notifications")
    
    # Manual notification
    st.subheader("📤 Send Custom Notification")
    
    with st.form("send_notification"):
        title = st.text_input("Title")
        message = st.text_area("Message")
        priority = st.selectbox("Priority", ["low", "medium", "high"])
        
        if st.form_submit_button("Send Notification"):
            result = api.post("/api/v1/notifications/", {
                "type": "custom",
                "title": title,
                "message": message,
                "priority": priority
            })
            
            if result.get("success"):
                st.success("✅ Notification sent!")
            else:
                st.error("❌ Failed to send notification")


def show_agents():
    """Agents status page"""
    st.title("⚙️ Agent Management")
    
    # Get agents status
    agents_status = api.get("/api/v1/agents/")
    
    if agents_status.get("success"):
        agents_data = agents_status.get("data", {})
        orchestrator_status = agents_data.get("orchestrator", {})
        agents = agents_data.get("agents", {})
        
        # Orchestrator status
        st.subheader("🎛️ Orchestrator Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", orchestrator_status.get("status", "unknown"))
        
        with col2:
            st.metric("Uptime (s)", orchestrator_status.get("uptime_seconds", 0))
        
        with col3:
            st.metric("Queue Size", orchestrator_status.get("queue_size", 0))
        
        # Individual agents
        st.subheader("🤖 Agent Status")
        
        for agent_name, agent_data in agents.items():
            with st.expander(f"🔧 {agent_name.title()} Agent"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {agent_data.get('status', 'unknown')}")
                    st.write(f"**Running:** {'✅' if agent_data.get('is_running') else '❌'}")
                    st.write(f"**Agent ID:** {agent_data.get('agent_id', 'N/A')}")
                
                with col2:
                    metrics = agent_data.get("metrics", {})
                    st.write(f"**Tasks Processed:** {metrics.get('tasks_processed', 0)}")
                    st.write(f"**Errors:** {metrics.get('errors_count', 0)}")
                    st.write(f"**Last Activity:** {metrics.get('last_activity', 'N/A')}")
    
    else:
        st.error("Failed to load agent status")


def show_settings():
    """Settings page"""
    st.title("🛠️ Settings")
    
    # API Configuration
    st.subheader("🔗 API Configuration")
    
    new_backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url
    )
    
    if st.button("Update Backend URL"):
        st.session_state.backend_url = new_backend_url
        st.success("Backend URL updated!")
        st.experimental_rerun()
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    system_info = api.get("/api/v1/status")
    if system_info:
        st.json(system_info)
    
    # Refresh data
    st.subheader("🔄 Data Management")
    
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    # Debug mode
    st.subheader("🐛 Debug Mode")
    
    debug_mode = st.checkbox("Enable Debug Mode")
    
    if debug_mode:
        st.write("Session State:")
        st.json(dict(st.session_state))


if __name__ == "__main__":
    main()