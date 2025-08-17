# Web Interface Integration Patterns

> 📱 **User-Friendly Interfaces**: Integrate agent systems with Streamlit, Gradio, and React for business user accessibility.

## Navigation
- **Previous**: [Dual Interface Design](dual_interface_design.md)
- **Next**: [Progress Tracking](progress_tracking.md)
- **Implementation**: [Level 2: Standard](../05_implementation_levels/level_2_standard.md) → [Level 3: Complex](../05_implementation_levels/level_3_complex.md)
- **Reference**: [Templates](../06_reference/templates.md) → [Troubleshooting](../06_reference/troubleshooting.md)

---

## Overview

Modern agent systems require user-friendly web interfaces for business users. This section provides proven patterns for integrating agent systems with web frameworks like Streamlit, Gradio, and React, with emphasis on real-time interactions and user experience.

---

## Framework Comparison

| Framework | Best For | Complexity | Time to MVP | Enterprise Ready |
|-----------|----------|------------|-------------|------------------|
| **Streamlit** | Business dashboards, data apps | Low | 1-2 hours | ✅ Yes |
| **Gradio** | ML demos, quick prototypes | Very Low | 30 minutes | ⚠️ Limited |
| **React + FastAPI** | Custom apps, enterprise | High | 1-2 weeks | ✅ Yes |

---

## Streamlit Integration Patterns

### Complete Streamlit Agent Interface

```python
import streamlit as st
import asyncio
from typing import Dict, Any, Optional
import time
import threading

class StreamlitAgentInterface:
    """Complete Streamlit integration for agent systems"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="AI Agent System Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UX
        st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stProgress > div > div > div > div {
            background-color: #00cc88;
        }
        .success-message {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 0.25rem;
            margin: 1rem 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #dee2e6;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'operation_history' not in st.session_state:
            st.session_state.operation_history = []
        
        if 'current_operation' not in st.session_state:
            st.session_state.current_operation = None
        
        if 'operation_results' not in st.session_state:
            st.session_state.operation_results = {}
        
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {
                'auto_refresh': True,
                'show_advanced': False,
                'theme': 'light'
            }
    
    def render_dashboard(self):
        """Render main dashboard interface"""
        st.title("🤖 AI Agent System Dashboard")
        
        # Sidebar navigation
        page = self.render_sidebar()
        
        # Main content area
        if page == "Dashboard":
            self.render_main_dashboard()
        elif page == "Generate Report":
            self.render_report_generation()
        elif page == "View History":
            self.render_operation_history()
        elif page == "System Status":
            self.render_system_status()
        elif page == "Settings":
            self.render_settings()
    
    def render_sidebar(self) -> str:
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # System status indicator
        system_status = self.get_system_status()
        status_color = "🟢" if system_status["healthy"] else "🔴"
        st.sidebar.markdown(f"{status_color} System Status: {'Healthy' if system_status['healthy'] else 'Issues'}")
        
        # Navigation menu
        pages = [
            "Dashboard",
            "Generate Report", 
            "View History",
            "System Status",
            "Settings"
        ]
        
        return st.sidebar.selectbox("Select Page", pages)
    
    def render_main_dashboard(self):
        """Render main dashboard with key metrics"""
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🚀 Active Operations", 
                value=len(st.session_state.get('active_operations', [])),
                delta="2 started today"
            )
        
        with col2:
            st.metric(
                label="✅ Completed Today", 
                value=len([op for op in st.session_state.operation_history 
                          if op.get('status') == 'completed']),
                delta="5 since yesterday"
            )
        
        with col3:
            st.metric(
                label="⚡ Avg Response Time", 
                value="2.3s",
                delta="-0.5s vs yesterday"
            )
        
        with col4:
            st.metric(
                label="🎯 Success Rate", 
                value="98.5%",
                delta="1.2%"
            )
        
        # Recent activity
        st.header("📈 Recent Activity")
        
        if st.session_state.operation_history:
            # Display recent operations in a table
            recent_ops = st.session_state.operation_history[-5:]
            
            for op in reversed(recent_ops):
                with st.expander(f"{op['type']} - {time.ctime(op['timestamp'])}"):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.text(f"Operation ID: {op['id']}")
                        st.text(f"Status: {op['status'].title()}")
                    
                    with col2:
                        if 'result' in op and 'execution_time' in op['result']:
                            st.text(f"Duration: {op['result']['execution_time']:.1f}s")
                    
                    with col3:
                        if st.button(f"View Details", key=f"view_{op['id']}"):
                            st.session_state.selected_operation = op['id']
        else:
            st.info("No operations performed yet. Start by generating a report!")
            
        # Quick actions
        st.header("🚀 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Generate New Report", type="primary"):
                st.session_state.current_page = "Generate Report"
                st.rerun()
        
        with col2:
            if st.button("🔍 View System Status"):
                st.session_state.current_page = "System Status"
                st.rerun()
        
        with col3:
            if st.button("📚 View History"):
                st.session_state.current_page = "View History"
                st.rerun()
    
    def render_report_generation(self):
        """Render report generation interface with real-time progress"""
        st.header("📊 Generate New Report")
        
        # Report configuration form
        with st.form("report_config"):
            col1, col2 = st.columns(2)
            
            with col1:
                month = st.selectbox(
                    "Select Month",
                    range(1, 13),
                    format_func=lambda x: time.strftime('%B', time.struct_time((0,x,0,0,0,0,0,0,0)))
                )
                
                include_charts = st.checkbox("Include Charts", value=True)
                
                advanced_options = st.checkbox("Show Advanced Options")
            
            with col2:
                year = st.selectbox("Select Year", range(2020, 2030), index=4)
                
                output_formats = st.multiselect(
                    "Output Formats",
                    ["PDF", "Excel", "CSV"],
                    default=["PDF", "Excel"]
                )
                
                priority = st.selectbox("Priority", ["Normal", "High", "Low"])
            
            # Advanced options (conditional)
            if advanced_options:
                st.subheader("Advanced Configuration")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    data_quality_threshold = st.slider("Data Quality Threshold", 0.0, 1.0, 0.95)
                    include_raw_data = st.checkbox("Include Raw Data")
                
                with col4:
                    max_execution_time = st.slider("Max Execution Time (minutes)", 1, 30, 10)
                    notification_email = st.text_input("Notification Email (optional)")
            
            submitted = st.form_submit_button("🚀 Generate Report", type="primary")
        
        # Report generation with progress tracking
        if submitted:
            if not output_formats:
                st.error("Please select at least one output format.")
                return
            
            config = {
                "month": month,
                "year": year,
                "include_charts": include_charts,
                "output_formats": output_formats,
                "priority": priority
            }
            
            if advanced_options:
                config.update({
                    "data_quality_threshold": data_quality_threshold,
                    "include_raw_data": include_raw_data,
                    "max_execution_time": max_execution_time,
                    "notification_email": notification_email
                })
            
            self.execute_report_generation(config)
    
    def execute_report_generation(self, config: Dict):
        """Execute report generation with real-time progress"""
        
        # Create progress containers
        progress_container = st.container()
        
        with progress_container:
            st.info(f"🔄 Generating report for {time.strftime('%B', time.struct_time((0,config['month'],0,0,0,0,0,0,0)))} {config['year']}")
            
            # Progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create operation tracking
            operation_id = f"report_{config['month']}_{config['year']}_{int(time.time())}"
            st.session_state.current_operation = operation_id
            
            # Execute workflow with progress tracking
            try:
                result = self.run_workflow_with_progress(
                    config, progress_bar, status_text
                )
                
                # Display results
                self.display_operation_results(result)
                
                # Store in history
                st.session_state.operation_history.append({
                    "id": operation_id,
                    "type": "Report Generation",
                    "timestamp": time.time(),
                    "status": "completed",
                    "result": result,
                    "config": config
                })
                
                # Success notification
                st.success("✅ Report generated successfully!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Report generation failed: {str(e)}")
                status_text.error(f"Error: {str(e)}")
                
                # Store failed operation in history
                st.session_state.operation_history.append({
                    "id": operation_id,
                    "type": "Report Generation",
                    "timestamp": time.time(),
                    "status": "failed",
                    "error": str(e),
                    "config": config
                })
    
    def run_workflow_with_progress(self, config: Dict, progress_bar, status_text) -> Dict:
        """Run workflow with Streamlit progress updates"""
        
        class StreamlitProgressHandler:
            def __init__(self, bar, status):
                self.bar = bar
                self.status = status
                self.start_time = time.time()
            
            def update(self, stage: str, progress: int, message: str = ""):
                self.bar.progress(progress / 100.0)
                elapsed = time.time() - self.start_time
                
                status_msg = f"""
                **{stage}** - {message}
                
                ⏱️ Elapsed: {elapsed:.1f}s | 📊 Progress: {progress}%
                🔄 Status: In Progress
                """
                self.status.markdown(status_msg)
                
                # Allow UI to update
                time.sleep(0.1)
        
        progress_handler = StreamlitProgressHandler(progress_bar, status_text)
        
        # Simulate workflow stages
        stages = [
            ("Initializing", "Setting up workflow parameters"),
            ("Data Extraction", "Extracting data from sources"),
            ("Data Processing", "Processing and validating data"),
            ("Quality Checks", "Performing data quality validation"),
            ("Calculations", "Performing financial calculations"),
            ("Report Generation", "Creating report documents"),
            ("Finalization", "Completing and packaging results")
        ]
        
        total_stages = len(stages)
        
        for i, (stage, message) in enumerate(stages):
            progress = int((i + 1) / total_stages * 100)
            progress_handler.update(stage, progress, message)
            
            # Simulate processing time (replace with actual agent calls)
            processing_time = 1.0 + (i * 0.5)  # Variable processing time
            time.sleep(processing_time)
        
        # Final completion
        final_status = f"""
        **Completed** - All stages finished successfully
        
        ⏱️ Total Time: {time.time() - progress_handler.start_time:.1f}s
        📊 Progress: 100%
        ✅ Status: Completed
        """
        status_text.markdown(final_status)
        
        # Return mock result (replace with actual orchestrator result)
        return {
            "status": "success",
            "files_generated": [
                f"report_{config['month']}_{config['year']}.{fmt.lower()}"
                for fmt in config['output_formats']
            ],
            "execution_time": time.time() - progress_handler.start_time,
            "records_processed": 1247,
            "data_quality_score": 0.96,
            "config": config
        }
    
    def display_operation_results(self, result: Dict):
        """Display operation results in user-friendly format"""
        if result["status"] == "success":
            st.success("✅ Report generated successfully!")
            
            # Results summary cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Files Generated", len(result["files_generated"]))
            
            with col2:
                st.metric("Records Processed", f"{result['records_processed']:,}")
            
            with col3:
                st.metric("Execution Time", f"{result['execution_time']:.1f}s")
            
            with col4:
                st.metric("Data Quality", f"{result.get('data_quality_score', 0.95):.1%}")
            
            # File download section
            st.subheader("📁 Generated Files")
            
            for file_name in result["files_generated"]:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file_name}")
                with col2:
                    # In real implementation, provide actual download
                    if st.button(f"📥 Download", key=f"download_{file_name}"):
                        st.info(f"Downloading {file_name}...")
                        # Implement actual download logic here
        else:
            st.error(f"❌ Operation failed: {result.get('error', 'Unknown error')}")
    
    def render_operation_history(self):
        """Render operation history with interactive elements"""
        st.header("📈 Operation History")
        
        if not st.session_state.operation_history:
            st.info("No operations performed yet.")
            return
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", ["All", "Completed", "Failed", "In Progress"])
        with col2:
            days_filter = st.selectbox("Show Last", ["All Time", "7 Days", "30 Days"])
        with col3:
            sort_order = st.selectbox("Sort By", ["Most Recent", "Oldest First", "Duration"])
        
        # Apply filters
        filtered_history = self.filter_operation_history(
            st.session_state.operation_history, status_filter, days_filter
        )
        
        # Display operations
        for operation in filtered_history:
            with st.expander(f"{operation['type']} - {time.ctime(operation['timestamp'])}"):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_icon = "✅" if operation['status'] == 'completed' else "❌" if operation['status'] == 'failed' else "🔄"
                    st.markdown(f"**Status:** {status_icon} {operation['status'].title()}")
                
                with col2:
                    st.markdown(f"**ID:** `{operation['id']}`")
                
                with col3:
                    if 'result' in operation and 'execution_time' in operation['result']:
                        duration = operation['result']['execution_time']
                        st.markdown(f"**Duration:** {duration:.1f}s")
                
                # Results details
                if operation['status'] == 'completed' and 'result' in operation:
                    result = operation['result']
                    
                    if 'files_generated' in result:
                        st.markdown("**Files Generated:**")
                        for file_name in result['files_generated']:
                            st.text(f"  📄 {file_name}")
                    
                    if 'records_processed' in result:
                        st.text(f"Records Processed: {result['records_processed']:,}")
                
                # Configuration details
                if 'config' in operation:
                    with st.expander("Configuration Details"):
                        st.json(operation['config'])
    
    def filter_operation_history(self, history, status_filter, days_filter):
        """Apply filters to operation history"""
        filtered = history.copy()
        
        # Status filter
        if status_filter != "All":
            filtered = [op for op in filtered if op['status'] == status_filter.lower()]
        
        # Time filter
        if days_filter != "All Time":
            days = int(days_filter.split()[0])
            cutoff_time = time.time() - (days * 24 * 3600)
            filtered = [op for op in filtered if op['timestamp'] >= cutoff_time]
        
        return list(reversed(filtered))  # Most recent first
    
    def render_system_status(self):
        """Render system status and health monitoring"""
        st.header("🔧 System Status")
        
        # System health overview
        system_status = self.get_system_status()
        
        if system_status["healthy"]:
            st.success("🟢 System is operating normally")
        else:
            st.error("🔴 System issues detected")
        
        # Agent status grid
        st.subheader("Agent Status")
        
        agents_status = self.get_agents_status()
        
        cols = st.columns(3)
        for i, (agent_name, status) in enumerate(agents_status.items()):
            with cols[i % 3]:
                status_icon = "🟢" if status["healthy"] else "🔴"
                st.metric(
                    label=f"{status_icon} {agent_name.replace('_', ' ').title()}",
                    value=status["status"],
                    delta=f"{status.get('success_rate', 0):.1%} success rate"
                )
        
        # System metrics
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Operations", 3)
        
        with col2:
            st.metric("Avg Response Time", "2.3s", delta="-0.2s")
        
        with col3:
            st.metric("Success Rate", "98.5%", delta="1.2%")
        
        with col4:
            st.metric("Uptime", "99.9%")
        
        # Recent system events
        st.subheader("Recent System Events")
        
        events = [
            {"time": "2 minutes ago", "type": "info", "message": "Agent WebExtractor completed task successfully"},
            {"time": "5 minutes ago", "type": "info", "message": "New report generation started"},
            {"time": "12 minutes ago", "type": "warning", "message": "High memory usage detected on DataProcessor"},
            {"time": "1 hour ago", "type": "info", "message": "System maintenance completed"}
        ]
        
        for event in events:
            icon = "ℹ️" if event["type"] == "info" else "⚠️" if event["type"] == "warning" else "❌"
            st.text(f"{icon} {event['time']}: {event['message']}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        # In real implementation, this would check actual system health
        return {
            "healthy": True,
            "last_check": time.time(),
            "agents_online": 6,
            "active_operations": 3,
            "memory_usage": 0.45,
            "cpu_usage": 0.23
        }
    
    def get_agents_status(self) -> Dict[str, Dict]:
        """Get individual agent status"""
        # Mock agent status for demonstration
        return {
            "web_extractor": {"healthy": True, "status": "Ready", "success_rate": 0.95},
            "email_processor": {"healthy": True, "status": "Ready", "success_rate": 0.98},
            "data_consolidator": {"healthy": True, "status": "Ready", "success_rate": 0.97},
            "financial_calculator": {"healthy": True, "status": "Ready", "success_rate": 0.99},
            "report_generator": {"healthy": True, "status": "Ready", "success_rate": 0.96},
            "database_manager": {"healthy": True, "status": "Ready", "success_rate": 0.99}
        }
    
    def render_settings(self):
        """Render system settings and configuration"""
        st.header("⚙️ Settings")
        
        # User preferences
        st.subheader("User Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_refresh = st.checkbox(
                "Auto-refresh dashboard", 
                value=st.session_state.user_preferences['auto_refresh']
            )
            show_advanced = st.checkbox(
                "Show advanced options", 
                value=st.session_state.user_preferences['show_advanced']
            )
        
        with col2:
            refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 30)
            max_history = st.slider("Max history items", 10, 100, 50)
        
        # Theme selection
        theme = st.selectbox(
            "Theme", 
            ["Light", "Dark", "Auto"],
            index=0 if st.session_state.user_preferences['theme'] == 'light' else 1
        )
        
        # Notification settings
        st.subheader("Notifications")
        
        col3, col4 = st.columns(2)
        
        with col3:
            email_notifications = st.checkbox("Email notifications")
            browser_notifications = st.checkbox("Browser notifications")
        
        with col4:
            notification_email = st.text_input("Notification email")
            notification_frequency = st.selectbox("Frequency", ["Immediate", "Hourly", "Daily"])
        
        # System configuration (read-only display)
        st.subheader("System Configuration")
        
        config_info = {
            "Database": "PostgreSQL (Connected)",
            "OpenAI Model": "gpt-4o",
            "Cache Backend": "Redis",
            "File Storage": "./reports/output",
            "Log Level": "INFO",
            "Version": "3.1.0"
        }
        
        for key, value in config_info.items():
            st.text(f"{key}: {value}")
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            # Update session state
            st.session_state.user_preferences.update({
                'auto_refresh': auto_refresh,
                'show_advanced': show_advanced,
                'theme': theme.lower()
            })
            
            st.success("Settings saved successfully!")
            time.sleep(1)
            st.rerun()

# Usage example
def launch_streamlit_interface(orchestrator):
    """Launch Streamlit interface"""
    interface = StreamlitAgentInterface(orchestrator)
    interface.render_dashboard()

if __name__ == "__main__":
    # This would be called from main.py
    from core.orchestrator import UniversalOrchestrator
    orchestrator = UniversalOrchestrator(agents={})
    launch_streamlit_interface(orchestrator)
```

---

## Gradio Integration Alternative

For rapid prototyping and ML-focused interfaces:

```python
import gradio as gr
from typing import Iterator

class GradioAgentInterface:
    """Gradio integration for rapid prototyping"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
    
    def create_interface(self) -> gr.Interface:
        """Create Gradio interface for agent system"""
        
        def generate_report_gradio(month: int, year: int, format_type: str) -> Iterator[str]:
            """Generator function for progress updates"""
            yield "🔄 Initializing report generation..."
            
            stages = [
                "📊 Extracting data from sources",
                "🔍 Processing and validating data", 
                "💰 Performing financial calculations",
                "📄 Generating report documents",
                "✅ Report generation completed"
            ]
            
            for i, stage in enumerate(stages):
                yield f"Progress: {((i+1)/len(stages)*100):.0f}% - {stage}"
                time.sleep(1.5)
            
            yield f"📁 Report generated: report_{month}_{year}.{format_type.lower()}"
        
        # Create interface
        interface = gr.Interface(
            fn=generate_report_gradio,
            inputs=[
                gr.Slider(1, 12, value=1, label="Month"),
                gr.Slider(2020, 2025, value=2024, label="Year"),
                gr.Dropdown(["PDF", "Excel", "CSV"], value="PDF", label="Format")
            ],
            outputs=gr.Textbox(label="Progress", lines=10),
            title="🤖 AI Agent Report Generator",
            description="Generate reports using AI agents with real-time progress tracking",
            live=False
        )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch Gradio interface"""
        interface = self.create_interface()
        return interface.launch(**kwargs)
```

---

## React + FastAPI Integration

For enterprise-grade custom interfaces:

```python
# FastAPI backend for React integration
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

class ReactAgentAPI:
    """FastAPI backend for React agent interface"""
    
    def __init__(self, orchestrator):
        self.app = FastAPI(title="Agent System API")
        self.orchestrator = orchestrator
        self.active_connections = []
        self.setup_middleware()
        self.setup_routes()
    
    def setup_middleware(self):
        """Configure CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000"],  # React dev server
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.post("/api/reports/generate")
        async def generate_report(request: dict):
            """Generate report with progress updates via WebSocket"""
            operation_id = f"report_{int(time.time())}"
            
            # Start background task
            asyncio.create_task(
                self.execute_report_workflow(operation_id, request)
            )
            
            return {"operation_id": operation_id, "status": "started"}
        
        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get system health status"""
            return {
                "healthy": True,
                "agents": 6,
                "active_operations": len(self.active_connections),
                "timestamp": time.time()
            }
        
        @self.app.websocket("/ws/progress/{operation_id}")
        async def websocket_progress(websocket: WebSocket, operation_id: str):
            """WebSocket for real-time progress updates"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await asyncio.sleep(1)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def execute_report_workflow(self, operation_id: str, config: dict):
        """Execute workflow and send progress via WebSocket"""
        
        # Find WebSocket connection for this operation
        websocket = None
        for conn in self.active_connections:
            # In real implementation, match by operation_id
            websocket = conn
            break
        
        if not websocket:
            return
        
        try:
            # Simulate workflow progress
            stages = [
                ("initialization", 10, "Initializing workflow"),
                ("data_extraction", 30, "Extracting data"),
                ("processing", 60, "Processing data"),
                ("calculation", 80, "Performing calculations"),
                ("generation", 95, "Generating report"),
                ("completion", 100, "Workflow completed")
            ]
            
            for stage, progress, message in stages:
                await websocket.send_json({
                    "operation_id": operation_id,
                    "stage": stage,
                    "progress": progress,
                    "message": message,
                    "timestamp": time.time()
                })
                
                await asyncio.sleep(2)  # Simulate processing time
                
        except Exception as e:
            await websocket.send_json({
                "operation_id": operation_id,
                "stage": "error",
                "progress": 0,
                "message": f"Error: {str(e)}",
                "timestamp": time.time()
            })
```

---

## Best Practices for Web Integration

1. **Async Compatibility**: Use async patterns throughout for responsiveness
2. **Progress Transparency**: Always show progress for long-running operations
3. **Error Handling**: Graceful error display with recovery suggestions
4. **State Management**: Proper session state handling across page refreshes
5. **Performance**: Optimize for real-time updates without blocking UI
6. **Accessibility**: Design with accessibility standards in mind
7. **Mobile Responsiveness**: Ensure interfaces work on mobile devices

---

## Interface Selection Guide

| Requirement | Streamlit | Gradio | React + FastAPI |
|------------|-----------|---------|-----------------|
| **Development Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Customization** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Enterprise Features** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Learning Curve** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Community Support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## Next Steps

- **Progress Tracking**: [Real-time Feedback Systems](progress_tracking.md)
- **Implementation**: [Level 2 Standard Systems](../05_implementation_levels/level_2_standard.md)
- **Dual Interface**: [Complete Interface Architecture](dual_interface_design.md)

---

*These web integration patterns enable rapid development of user-friendly interfaces while maintaining the power and flexibility of your agent systems.*