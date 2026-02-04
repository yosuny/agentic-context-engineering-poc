import streamlit as st
import time
import json
import os
import sys
import tempfile
import atexit
import shutil

# Create a temporary directory for logs to satisfy ACE's logging requirements
TEMP_LOG_DIR = tempfile.mkdtemp(prefix="ace_mock_logs_")

# Cleanup on exit
def cleanup_temp_dir():
    try:
        shutil.rmtree(TEMP_LOG_DIR)
    except:
        pass

atexit.register(cleanup_temp_dir)

# Add project root to path to import ace modules
# The structure is:
# project_root/ace/ (contains utils.py, playbook_utils.py, logger.py)
# project_root/ace/ace/ (contains ace.py, __init__.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ace_source_dir = os.path.join(project_root, 'ace')
sys.path.append(ace_source_dir)

# Import utils directly since it's in the ace_source_dir
import utils
from ace.ace import ACE
from playbook_utils import update_bullet_counts

# Mock Client Classes
class MockResponse:
    def __init__(self, content):
        self.choices = [type('obj', (object,), {'message': type('obj', (object,), {'content': content})})]
        self.usage = type('obj', (object,), {'prompt_tokens': 100, 'completion_tokens': 50})

class MockClient:
    def __init__(self, role):
        self.role = role
        self.chat = type('obj', (object,), {'completions': self})

    def create(self, model, messages, **kwargs):
        prompt = messages[0]['content']
        
        # Generator Mock Logic
        if self.role == 'generator':
            time.sleep(1) # Simulate think time
            return MockResponse(json.dumps({
                "final_answer": "42.00",
                "reasoning": "Based on the provided context and playbook strategies, I have calculated the answer."
            }))
            
        # Reflector Mock Logic
        elif self.role == 'reflector':
            time.sleep(1)
            # 50% chance of helpful, 50% chance of harmful for demo
            return MockResponse(json.dumps({
                "reflection": "The generator followed the strategy correctly.",
                "helpful_bullets": ["fin-00001", "cal-00003"],
                "harmful_bullets": []
            }))

        # Curator Mock Logic
        elif self.role == 'curator':
            time.sleep(1.5)
            # Predefined mock strategies to show variety
            strategies = [
                ("common_mistakes_to_avoid", "Use precise decimal points for financial calculations."),
                ("strategies_and_insights", "Double-check the currency unit (e.g., USD vs KRW) in reports."),
                ("common_mistakes_to_avoid", "Do not confuse Net Income with Operating Income."),
                ("strategies_and_insights", "Always cross-reference dates with the fiscal year calendar.")
            ]
            
            # Select a strategy based on time or random (using simple hash of time for variety)
            idx = int(time.time()) % len(strategies)
            section, content = strategies[idx]
            
            new_strategy = f"{content} (Curated at {time.strftime('%H:%M:%S')})"
            
            return MockResponse(json.dumps({
                "reasoning": f"Observed pattern of errors related to {content.split()[0:2]}.",
                "operations": [
                    {"type": "ADD", "section": section, "content": new_strategy}
                ]
            }))
            
        return MockResponse("Mock response")

# Monkeypatch initialize_clients to return mock clients
def mock_initialize_clients(api_provider):
    return MockClient('generator'), MockClient('reflector'), MockClient('curator')

# Inject mock function
utils.initialize_clients = mock_initialize_clients

# IMPORTANT: ace.ace imports everything from utils using 'from utils import *'
# So we must also patch the function in the ace.ace namespace
import ace.ace
ace.ace.initialize_clients = mock_initialize_clients
# Also patch llm.py's client usage if needed, but ACE class takes clients from initialize_clients so this might be enough.
# However, ACE.__init__ calls initialize_clients from utils.
# Let's also patch the specific check in llm.py if it validates client types, but llm.py uses duck typing.

# Streamlit UI
st.set_page_config(page_title="ACE Mock Test Dashboard", layout="wide")

st.title("🤖 ACE Framework Mock Test Dashboard")
st.markdown("Run the ACE framework with **Mock Agents** to visualize the Generator -> Reflector -> Curator pipeline.")

# Initialize Session State
if 'ace_system' not in st.session_state:
    st.session_state.ace_system = None
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'playbook_history' not in st.session_state:
    st.session_state.playbook_history = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

# Sidebar Config
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["offline", "online"])
    num_epochs = st.number_input("Epochs", 1, 10, 1)
    
    if st.button("Initialize System"):
        # Reset state
        st.session_state.logs = []
        st.session_state.current_step = 0
        
        # Initialize ACE
        ace = ACE(
            api_provider="mock",
            generator_model="mock-gpt",
            reflector_model="mock-gpt",
            curator_model="mock-gpt",
            initial_playbook="## STRATEGIES & INSIGHTS\n[fin-00001] helpful=0 harmful=0 :: Always verify currency.\n\n## COMMON MISTAKES TO AVOID\n"
        )
        st.session_state.ace_system = ace
        st.session_state.playbook_history = [ace.playbook]
        st.success("System Initialized with Mock Clients!")

# Main Layout
col1, col2 = st.columns([1, 1])

# Left Column: Process Flow (Simulation)
with col1:
    st.subheader("📝 Process Flow")
    
    # Custom logger to capture prints/logs -> session state
    # This is a bit tricky since ACE uses print(). 
    # For this mock app, we will simulate the flow manually utilizing the ACE components 
    # rather than calling ace.run() which blocks.
    
    if st.session_state.ace_system:
        # Step Request
        if st.button("▶ Run Next Step (Single Mock Sample)"):
            ace = st.session_state.ace_system
            step = st.session_state.current_step + 1
            st.session_state.current_step = step
            
            # 1. Generator
            with st.status(f"Step {step}: Generating Answer...", expanded=True) as status:
                st.write("🔵 **Generator** is thinking...")
                q = "What is the Net Asset Value?"
                context = "Assets: 100, Liabilities: 58."
                
                gen_response, bullet_ids, _ = ace.generator.generate(q, ace.playbook, context, "", False, "mock_call")
                final_answer = json.loads(gen_response)['final_answer']
                st.write(f"👉 Answer: `{final_answer}`")
                st.session_state.logs.append(f"Answer generated: {final_answer}")
                
                # 2. Reflector
                st.write("🟡 **Reflector** is analyzing...")
                reflection_res, tags, _ = ace.reflector.reflect(q, gen_response, final_answer, "42.00", "diff", "", True, False, "mock_ref")
                st.write(f"👉 Reflection: {reflection_res}")
                
                # Update counts
                if tags:
                    old_pb = ace.playbook
                    ace.playbook = update_bullet_counts(ace.playbook, tags)
                    
                    # Log which ones were updated
                    updated_ids = [t.get('id') or t.get('bullet') for t in tags]
                    st.write(f"📈 **Updated points** for: `{', '.join(updated_ids)}`")
                    st.session_state.logs.append(f"Counters updated for {len(tags)} bullets.")
                
                # 3. Curator (Trigger every step for demo)
                st.write("🟢 **Curator** is updating playbook...")
                # curate(..., log_dir=TEMP_LOG_DIR, ...)
                ace.playbook, _, ops, _ = ace.curator.curate(ace.playbook, reflection_res, context, step, 100, 1000, {}, True, False, "mock_cur", TEMP_LOG_DIR, ace.next_global_id)
                
                if ops:
                    st.write("✨ **Update Applied!**")
                    for op in ops:
                        st.json(op)
                else:
                    st.write("Taking a break (No changes).")
                    
                status.update(label=f"Step {step} Complete!", state="complete")
                
            # Save playbook state
            st.session_state.playbook_history.append(ace.playbook)

    # Display Logs
    for log in reversed(st.session_state.logs):
        st.text(f"> {log}")

# Right Column: Playbook State
with col2:
    st.subheader("📚 Live Playbook")
    if st.session_state.ace_system:
        current_pb = st.session_state.ace_system.playbook
        st.text_area("Current Content", current_pb, height=600)
    else:
        st.info("Initialize the system to see the playbook.")
