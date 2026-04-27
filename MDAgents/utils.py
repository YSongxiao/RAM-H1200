import os
import json
import random
import base64
from contextlib import redirect_stdout
from io import BytesIO, StringIO
import mimetypes
import re
import time
import requests
from PIL import Image
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
from openai import OpenAI
from pptree import *

try:
    import google.generativeai as genai
except ImportError:
    genai = None

def get_openai_api_key():
    api_key = os.environ.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise EnvironmentError("Set openai_api_key or OPENAI_API_KEY before calling OpenAI models.")
    if api_key == "你的_api_key":
        raise EnvironmentError('Replace the placeholder OPENAI_API_KEY="你的_api_key" with your real OpenAI API key.')
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EnvironmentError("OPENAI_API_KEY must contain only ASCII characters. Please export the real API key, not a placeholder.") from exc
    return api_key

CONCISE_SYSTEM_RULE = (
    "Keep responses concise. Use the minimum text needed. "
    "Do not provide long step-by-step analysis unless explicitly required."
)

CONCISE_USER_RULE = (
    "Keep the answer brief. No long analysis. Use the minimum text needed."
)

SHORT_REASON_RULE = (
    "If a reason is requested, keep it to one short sentence or short phrase."
)

def concise_system(prompt):
    return f"{prompt} {CONCISE_SYSTEM_RULE}"

def concise_user(prompt, extra=None):
    suffix = CONCISE_USER_RULE if extra is None else f"{CONCISE_USER_RULE} {extra}"
    return f"{prompt}\n\n{suffix}"

class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-4o-mini', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.trace = []
        self._seen_image_paths = set()

        if self.model_info == 'gemini-pro':
            if genai is None:
                raise ImportError("google.generativeai is required when using gemini-pro.")
            self.model = genai.GenerativeModel('gemini-pro')
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.client = None
            self.last_time_ms = 0
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})

    def _openai_model_name(self):
        if self.model_info == 'gpt-3.5':
            return "gpt-3.5-turbo"
        return self.model_info

    def _build_user_content(self, message, img_path=None):
        if img_path is None:
            return message
        normalized_img_path = os.path.abspath(os.path.expanduser(img_path))
        if normalized_img_path in self._seen_image_paths:
            return message
        self._seen_image_paths.add(normalized_img_path)
        return [
            {"type": "text", "text": message},
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(normalized_img_path)},
            },
        ]

    def _trace_content(self, message, img_path=None):
        return {
            "text": message,
            "img_path": img_path,
        }

    def export_trace(self):
        return {
            "role": self.role,
            "model": self.model_info,
            "instruction": self.instruction,
            "calls": list(self.trace),
        }

    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info == 'gemini-pro':
            for _ in range(10):
                try:
                    response = self._chat.send_message(message, stream=True)
                    responses = ""
                    for chunk in response:
                        responses += chunk.text + "\n"
                    self.trace.append({
                        "method": "chat",
                        "prompt": self._trace_content(message, img_path),
                        "response": responses,
                        "elapsed_ms": None,
                    })
                    return responses
                except:
                    continue
            return "Error: Failed to get response from Gemini."

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:
            self.messages.append({"role": "user", "content": self._build_user_content(message, img_path)})

            response_content, elapsed_ms = call_openai_messages(
                model=self._openai_model_name(),
                messages=self.messages,
            )

            self.last_time_ms = elapsed_ms
            self.messages.append({"role": "assistant", "content": response_content})
            self.trace.append({
                "method": "chat",
                "prompt": self._trace_content(message, img_path),
                "response": response_content,
                "elapsed_ms": elapsed_ms,
            })
            return response_content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini']:      
            self.messages.append({"role": "user", "content": self._build_user_content(message, img_path)})
            
            temperatures = [0.0]
            
            responses = {}
            trace_responses = []
            for temperature in temperatures:
                response_content, elapsed_ms = call_openai_messages(
                    model=self._openai_model_name(),
                    messages=self.messages,
                    temperature=temperature,
                )
                
                self.last_time_ms = elapsed_ms
                responses[temperature] = response_content
                trace_responses.append({
                    "temperature": temperature,
                    "response": response_content,
                    "elapsed_ms": elapsed_ms,
                })

            self.trace.append({
                "method": "temp_responses",
                "prompt": self._trace_content(message, img_path),
                "responses": trace_responses,
            })
                
            return responses
        
        elif self.model_info == 'gemini-pro':
            response = self._chat.send_message(message, stream=True)
            responses = ""
            for chunk in response:
                responses += chunk.text + "\n"
            self.trace.append({
                "method": "temp_responses",
                "prompt": self._trace_content(message, img_path),
                "responses": [{
                    "temperature": None,
                    "response": responses,
                    "elapsed_ms": None,
                }],
            })
            return responses

class Group:
    def __init__(self, goal, members, question, examplers=None, img_path=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            member_prompt = concise_system(
                'You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower())
            )
            _agent = Agent(member_prompt, role=member_info['role'], model_info='gpt-4o-mini')
            _agent.chat(member_prompt)
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers
        self.img_path = img_path

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            for a_mem in assist_members:
                delivery_prompt += "\n{}".format(a_mem.role)
            
            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {}".format(self.question)
            delivery_prompt = concise_user(delivery_prompt, "Keep the investigation requests to short phrases.")
            try:
                delivery = lead_member.chat(delivery_prompt, img_path=self.img_path)
            except:
                delivery = assist_members[0].chat(delivery_prompt, img_path=self.img_path)

            investigations = []
            for a_mem in assist_members:
                investigation = a_mem.chat(
                    concise_user(
                        "You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery),
                        "Keep the summary very short."
                    ),
                    img_path=self.img_path
                )
                investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            for investigation in investigations:
                gathered_investigation += "[{}]\n{}\n".format(investigation[0], investigation[1])

            if self.examplers is not None:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n{self.examplers}\nQuestion: {self.question}"""
            else:
                investigation_prompt = f"""The gathered investigation from your asssitant clinicians is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            investigation_prompt = concise_user(investigation_prompt, SHORT_REASON_RULE)

            response = lead_member.chat(investigation_prompt, img_path=self.img_path)

            return response

        elif comm_type == 'external':
            return

def _canonical_agent_name(name):
    if name is None:
        return None
    cleaned = re.sub(r'^\d+[\.\)]\s*', '', str(name)).strip()
    cleaned = cleaned.split(' - ', 1)[0].strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.lower() if cleaned else None

def _extract_hierarchy_parent(hierarchy):
    if hierarchy is None:
        return None, None

    raw = str(hierarchy).strip()
    if not raw:
        return None, "Empty hierarchy string."

    if 'independent' in raw.lower():
        return None, None

    cleaned = re.sub(r'(?i)^hierarchy\s*:\s*', '', raw).strip()

    forward_parts = re.split(r'\s*(?:->|=>|>|→)\s*', cleaned, maxsplit=1)
    if len(forward_parts) == 2:
        parent_segment = forward_parts[0]
    else:
        backward_parts = re.split(r'\s*(?:<-|<=|<|←)\s*', cleaned, maxsplit=1)
        if len(backward_parts) == 2:
            parent_segment = backward_parts[1]
        else:
            parent_segment = cleaned

    candidates = [
        segment.strip(" .")
        for segment in re.split(r'\s*(?:==|=|,|/|;|\band\b|\bthen\b)\s*', parent_segment)
        if segment.strip(" .")
    ]
    parent = candidates[-1] if candidates else None
    parent = re.sub(r'^\d+[\.\)]\s*', '', parent).strip() if parent else None

    if not parent:
        return None, f"Unparseable hierarchy string: {raw}"
    return parent, None

def render_tree(node):
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_tree(node, horizontal=False)
    return buffer.getvalue().rstrip()

def parse_hierarchy(info, emojis, return_details=False):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    hierarchy_details = []
    pending_agents = []

    for count, (expert, hierarchy) in enumerate(info):
        try:
            expert_name = expert.split(' - ', 1)[0].split('.')[1].strip()
        except:
            expert_name = expert.split(' - ', 1)[0].strip()

        raw_hierarchy = hierarchy if hierarchy is not None else 'Independent'
        parent_name, warning = _extract_hierarchy_parent(raw_hierarchy)
        expert_key = _canonical_agent_name(expert_name)

        hierarchy_details.append({
            "expert": expert_name,
            "raw_hierarchy": raw_hierarchy,
            "parsed_parent": parent_name,
            "warning": warning,
        })
        pending_agents.append({
            "expert": expert_name,
            "expert_key": expert_key,
            "emoji": emojis[count],
            "parent_name": parent_name,
        })

    created_nodes = {"moderator": moderator}
    unresolved = list(enumerate(pending_agents))

    while unresolved:
        progress = False
        next_unresolved = []
        for index, agent_info in unresolved:
            detail = hierarchy_details[index]
            parent_key = _canonical_agent_name(agent_info["parent_name"]) if agent_info["parent_name"] else None
            if parent_key and parent_key == agent_info["expert_key"]:
                detail["warning"] = "Self-referential hierarchy detected; attached to moderator."
                parent_key = None

            parent_node = moderator if parent_key is None else created_nodes.get(parent_key)
            if parent_node is None:
                next_unresolved.append((index, agent_info))
                continue

            child_node = Node("{} ({})".format(agent_info["expert"], agent_info["emoji"]), parent_node)
            agents.append(child_node)
            created_nodes[agent_info["expert_key"]] = child_node
            detail["attached_to"] = parent_node.name.split("(")[0].strip()
            progress = True

        if progress:
            unresolved = next_unresolved
            continue

        for index, agent_info in next_unresolved:
            detail = hierarchy_details[index]
            if not detail.get("warning"):
                detail["warning"] = (
                    f"Parent '{agent_info['parent_name']}' was not found in recruited experts; "
                    "attached to moderator."
                )
            child_node = Node("{} ({})".format(agent_info["expert"], agent_info["emoji"]), moderator)
            agents.append(child_node)
            created_nodes[agent_info["expert_key"]] = child_node
            detail["attached_to"] = "moderator"
        break

    if return_details:
        return agents, hierarchy_details
    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        if genai is None:
            raise ImportError("google.generativeai is required when using a Gemini model.")
        genai.configure(api_key=os.environ['genai_api_key'])
        return genai, None
    elif 'gpt' in model_name:
        return None, None
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def image_to_data_url(img_path):
    mime_type = mimetypes.guess_type(img_path)[0] or "image/png"
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        return image_to_png_data_url(img_path)
    with open(img_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def image_to_png_data_url(img_path):
    with Image.open(img_path) as image:
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def load_image_binary_data(image_dir):
    samples = []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    for label in ["0", "1"]:
        label_dir = os.path.join(image_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for root, _, files in os.walk(label_dir):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext in image_exts:
                    samples.append({
                        "image_path": os.path.join(root, filename),
                        "label": label,
                    })
    return samples

def load_svdh_be_data(svdh_root, split="test", sites=None):
    svdh_root = os.path.expanduser(svdh_root)
    split_dir = os.path.join(svdh_root, split)
    annotation_path = os.path.join(split_dir, "_annotation_be_scores.json")
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with open(annotation_path, "r") as file:
        annotations = json.load(file)

    requested_sites = set(sites) if sites else None
    samples = []
    for image_name in sorted(annotations):
        patch_dir = os.path.join(split_dir, os.path.splitext(image_name)[0])
        if not os.path.isdir(patch_dir):
            raise FileNotFoundError(f"Patch folder not found: {patch_dir}")

        for score_key, label in sorted(annotations[image_name].items()):
            site = score_key.replace("BE_", "", 1)
            if requested_sites and site not in requested_sites:
                continue

            patch_path = os.path.join(patch_dir, f"{site}.bmp")
            if not os.path.isfile(patch_path):
                raise FileNotFoundError(f"Patch file not found for {image_name} {site}: {patch_path}")

            samples.append({
                "split": split,
                "image_name": image_name,
                "patch_dir": patch_dir,
                "site": site,
                "score_key": score_key,
                "patch_path": patch_path,
                "label": int(label),
            })

    return samples

def create_svdh_be_question(sample):
    return (
        "Task: Sharp/van der Heijde (SvdH) bone erosion scoring for rheumatoid arthritis hand/wrist X-ray patches.\n"
        f"Image: {sample['image_name']}; local anatomical site: {sample['site']}.\n"
        "Score this single cropped patch for bone erosion only. Do not score joint space narrowing, "
        "osteophytes, soft tissue swelling, or global disease activity.\n"
        "Options: (0) no visible erosion (1) minimal erosion (2) mild erosion "
        "(3) moderate erosion (4) marked erosion (5) severe erosion.\n"
        "Return the final answer as JSON in this exact schema: {\"score\":0,\"reason\":\"short visual reason\"}.\n"
        "Keep the reason extremely short. Do not provide long analysis."
    )

def extract_binary_label(response):
    match = re.search(r'"prediction"\s*:\s*"?([01])"?', response)
    if match:
        return match.group(1)
    match = re.search(r'\b([01])\b', response)
    if match:
        return match.group(1)
    return None

def extract_svdh_score(response):
    match = re.search(r'"score"\s*:\s*"?([0-5])"?', response)
    if match:
        return int(match.group(1))
    match = re.search(r'\b(?:score|answer)\s*[:=]?\s*([0-5])\b', response, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def response_to_text(response):
    if isinstance(response, dict):
        return "\n".join(response_to_text(value) for value in response.values())
    if isinstance(response, (list, tuple)):
        return "\n".join(response_to_text(value) for value in response)
    return str(response)

def parse_recruited_experts(recruited, num_agents):
    fallback = [
        ("1. Musculoskeletal Radiologist - Specializes in interpreting hand and wrist X-ray patches for erosive changes.", "Independent"),
        ("2. Rheumatologist - Focuses on rheumatoid arthritis and Sharp/van der Heijde scoring.", "Independent"),
        ("3. Orthopedic Hand Specialist - Specializes in hand and wrist bone and joint anatomy.", "Independent"),
        ("4. Musculoskeletal Imaging Researcher - Focuses on imaging biomarkers and scoring reliability.", "Independent"),
        ("5. Clinical Rheumatology Specialist - Provides disease-specific clinical scoring context.", "Independent"),
    ]

    agents_data = []
    for line in recruited.split('\n'):
        line = line.strip().lstrip("-*").strip()
        if not line or "-" not in line:
            continue

        if " - Hierarchy:" in line:
            main, hierarchy = line.split(" - Hierarchy:", 1)
            hierarchy = hierarchy.strip() or "Independent"
        else:
            main, hierarchy = line, "Independent"

        match = re.match(r"^(\d+)[\.\)]\s*(.+)$", main)
        if match:
            index, rest = match.groups()
        else:
            index, rest = str(len(agents_data) + 1), main

        role_description = re.split(r"\s+-\s+", rest, maxsplit=1)
        role = role_description[0].strip()
        description = role_description[1].strip() if len(role_description) > 1 else "provides relevant medical expertise."
        if not role:
            continue

        agents_data.append((f"{index}. {role} - {description}", hierarchy))
        if len(agents_data) == num_agents:
            break

    if len(agents_data) < num_agents:
        existing_roles = {agent[0].split(" - ")[0].split(".", 1)[-1].strip().lower() for agent in agents_data}
        for agent in fallback:
            role = agent[0].split(" - ")[0].split(".", 1)[-1].strip().lower()
            if role not in existing_roles:
                _, description = agent[0].split(" - ", 1)
                agents_data.append((f"{len(agents_data) + 1}. {role.title()} - {description}", agent[1]))
            if len(agents_data) == num_agents:
                break

    return agents_data

def process_image_binary_query(img_path, model):
    question = (
        "Diagnose the medical image as a binary classification task.\n"
        "Return 1 if the image indicates disease/abnormality, and 0 if it indicates no disease/normal.\n"
        "Respond only as JSON in this exact schema: {\"prediction\":\"0 or 1\",\"reason\":\"short visual reason\"}.\n"
        "Keep the reason extremely short. Do not provide long analysis."
    )
    response, elapsed_ms = call_openai_image_query(
        model=model,
        img_path=img_path,
        system_prompt=(
            "You are a cautious medical imaging assistant. You can inspect the image and choose "
            "only between 1=disease/abnormality and 0=no disease/normal. "
            + CONCISE_SYSTEM_RULE
        ),
        user_prompt=question,
    )
    return {
        "prediction": extract_binary_label(response),
        "response": response,
        "time_ms": elapsed_ms,
    }

def process_svdh_be_mdagents_query(sample, model, difficulty="adaptive"):
    class SvdHArgs:
        dataset = "svdh_be"
        task = "svdh_be"

    question = create_svdh_be_question(sample)
    start_time = time.perf_counter()
    selected_difficulty, difficulty_trace = determine_difficulty(question, difficulty, return_trace=True)

    if selected_difficulty == 'basic':
        response, workflow_trace = process_basic_query(
            question, [], model, SvdHArgs(), img_path=sample["patch_path"], return_trace=True
        )
    elif selected_difficulty == 'intermediate':
        response, workflow_trace = process_intermediate_query(
            question, [], model, SvdHArgs(), img_path=sample["patch_path"], return_trace=True
        )
    elif selected_difficulty == 'advanced':
        response = process_advanced_query(question, model, SvdHArgs(), img_path=sample["patch_path"])
        workflow_trace = {
            "mode": "advanced",
            "note": "Detailed trace capture is currently implemented for basic and intermediate workflows.",
        }
    else:
        raise ValueError(f"Unsupported SvdH difficulty: {selected_difficulty}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    response_text = response_to_text(response)
    return {
        "difficulty": selected_difficulty,
        "question": question,
        "prediction": extract_svdh_score(response_text),
        "response": response,
        "time_ms": elapsed_ms,
        "trace": {
            "difficulty_selection": difficulty_trace,
            "workflow": workflow_trace,
        },
    }

def compute_svdh_metrics(results):
    scored = [result for result in results if result.get("prediction") is not None]
    if not scored:
        return {
            "num_scored": 0,
            "qwk": None,
            "mae": None,
            "bacc_percent": 0,
            "acc_percent": 0,
            "w1_acc_percent": 0,
            "p_n_sen_percent": 0,
            "p_n_acc_percent": 0,
            "params_m": None,
            "time_ms": None,
            "exact_accuracy": 0,
            "within_1_accuracy": 0,
            "per_site": {},
        }

    exact = sum(1 for result in scored if result["prediction"] == result["label"])
    abs_errors = [abs(result["prediction"] - result["label"]) for result in scored]
    labels = [result["label"] for result in scored]
    predictions = [result["prediction"] for result in scored]
    time_values = [result["time_ms"] for result in scored if result.get("time_ms") is not None]
    acc = exact / len(scored)
    within_1_acc = sum(1 for error in abs_errors if error <= 1) / len(scored)
    per_site = {}
    for result in scored:
        site = result["site"]
        per_site.setdefault(site, {"count": 0, "exact": 0, "abs_error": 0})
        per_site[site]["count"] += 1
        per_site[site]["exact"] += int(result["prediction"] == result["label"])
        per_site[site]["abs_error"] += abs(result["prediction"] - result["label"])

    for site, values in per_site.items():
        values["exact_accuracy"] = values["exact"] / values["count"]
        values["mae"] = values["abs_error"] / values["count"]

    return {
        "num_scored": len(scored),
        "qwk": quadratic_weighted_kappa(labels, predictions, min_rating=0, max_rating=5),
        "mae": sum(abs_errors) / len(scored),
        "bacc_percent": balanced_accuracy_percent(labels, predictions, classes=range(6)),
        "acc_percent": acc * 100,
        "w1_acc_percent": within_1_acc * 100,
        "p_n_sen_percent": positive_negative_sensitivity_percent(labels, predictions),
        "p_n_acc_percent": positive_negative_accuracy_percent(labels, predictions),
        "params_m": None,
        "time_ms": sum(time_values) / len(time_values) if time_values else None,
        "exact_accuracy": acc,
        "within_1_accuracy": within_1_acc,
        "per_site": per_site,
    }

def quadratic_weighted_kappa(labels, predictions, min_rating=0, max_rating=5):
    num_ratings = max_rating - min_rating + 1
    observed = [[0 for _ in range(num_ratings)] for _ in range(num_ratings)]
    actual_hist = [0 for _ in range(num_ratings)]
    pred_hist = [0 for _ in range(num_ratings)]

    for label, prediction in zip(labels, predictions):
        if label < min_rating or label > max_rating or prediction < min_rating or prediction > max_rating:
            continue
        label_idx = label - min_rating
        pred_idx = prediction - min_rating
        observed[label_idx][pred_idx] += 1
        actual_hist[label_idx] += 1
        pred_hist[pred_idx] += 1

    total = sum(actual_hist)
    if total == 0:
        return None

    numerator = 0
    denominator = 0
    max_distance = (num_ratings - 1) ** 2
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight = ((i - j) ** 2) / max_distance if max_distance else 0
            expected = actual_hist[i] * pred_hist[j] / total
            numerator += weight * observed[i][j]
            denominator += weight * expected

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return 1 - numerator / denominator

def balanced_accuracy_percent(labels, predictions, classes):
    recalls = []
    for cls in classes:
        support = sum(1 for label in labels if label == cls)
        if support == 0:
            continue
        true_positive = sum(1 for label, prediction in zip(labels, predictions) if label == cls and prediction == cls)
        recalls.append(true_positive / support)
    return (sum(recalls) / len(recalls) * 100) if recalls else 0

def positive_negative_sensitivity_percent(labels, predictions):
    binary_labels = [int(label > 0) for label in labels]
    binary_predictions = [int(prediction > 0) for prediction in predictions]
    positive_total = sum(binary_labels)
    negative_total = len(binary_labels) - positive_total
    sensitivities = []

    if positive_total:
        positive_hits = sum(1 for label, prediction in zip(binary_labels, binary_predictions) if label == 1 and prediction == 1)
        sensitivities.append(positive_hits / positive_total)
    if negative_total:
        negative_hits = sum(1 for label, prediction in zip(binary_labels, binary_predictions) if label == 0 and prediction == 0)
        sensitivities.append(negative_hits / negative_total)

    return (sum(sensitivities) / len(sensitivities) * 100) if sensitivities else 0

def positive_negative_accuracy_percent(labels, predictions):
    binary_labels = [int(label > 0) for label in labels]
    binary_predictions = [int(prediction > 0) for prediction in predictions]
    if not binary_labels:
        return 0
    correct = sum(1 for label, prediction in zip(binary_labels, binary_predictions) if label == prediction)
    return correct / len(binary_labels) * 100

def call_openai_messages(model, messages, temperature=None):
    payload = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Failed to serialize OpenAI payload to valid JSON: {exc}") from exc

    start_time = time.perf_counter()
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json",
        },
        data=payload_json.encode("utf-8"),
        timeout=120,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason}: {response.text} "
            f"(model={model}, payload_bytes={len(payload_json.encode('utf-8'))})",
            response=response,
        )
    return response.json()["choices"][0]["message"]["content"], elapsed_ms

def call_openai_image_query(model, img_path, system_prompt, user_prompt):
    if model == 'gpt-3.5':
        raise ValueError("gpt-3.5 does not support image input. Use gpt-4o-mini or gpt-4o.")

    return call_openai_messages(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img_path)},
                    },
                ],
            },
        ],
        temperature=0,
    )

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'../data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'../data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty, return_trace=False):
    trace = {
        "requested_mode": difficulty,
        "question": question,
    }
    if difficulty != 'adaptive':
        trace["selection_mode"] = "fixed"
        trace["selected_difficulty"] = difficulty
        if return_trace:
            return difficulty, trace
        return difficulty
    
    difficulty_prompt = concise_user(
        f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision.""",
        "Return only one word: basic, intermediate, or advanced."
    )
    
    medical_agent = Agent(instruction=concise_system('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.'), role='medical expert', model_info='gpt-3.5')
    medical_agent.chat(concise_system('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.'))
    response = medical_agent.chat(difficulty_prompt)

    selected_difficulty = None
    if 'basic' in response.lower() or '1)' in response.lower():
        selected_difficulty = 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        selected_difficulty = 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        selected_difficulty = 'advanced'

    trace.update({
        "selection_mode": "adaptive",
        "system_prompt": medical_agent.instruction,
        "difficulty_prompt": difficulty_prompt,
        "model_response": response,
        "selected_difficulty": selected_difficulty,
        "agent_trace": medical_agent.export_trace(),
    })

    if return_trace:
        return selected_difficulty, trace
    return selected_difficulty

def process_basic_query(question, examplers, model, args, img_path=None, return_trace=False):
    medical_agent = Agent(instruction=concise_system('You are a helpful medical agent.'), role='medical expert', model_info=model)
    new_examplers = []
    trace = {
        "mode": "basic",
        "question": question,
        "img_path": img_path,
        "fewshot_examplers": [],
    }
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(
                concise_user(
                    f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}",
                    "Use only one short sentence."
                )
            )

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
            trace["fewshot_examplers"].append(dict(tmp_exampler))
    
    single_agent = Agent(instruction=concise_system('You are a helpful assistant that answers multiple choice questions about medical knowledge.'), role='medical expert', examplers=new_examplers, model_info=model)
    single_agent.chat(concise_system('You are a helpful assistant that answers multiple choice questions about medical knowledge.'))
    if getattr(args, 'task', None) == 'svdh_be':
        final_prompt = concise_user(
            f'''The following is a medical image scoring question. Inspect the attached patch and follow the requested output schema.\n\n**Question:** {question}\nAnswer: ''',
            SHORT_REASON_RULE
        )
    else:
        final_prompt = concise_user(
            f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''',
            "Keep the explanation very short."
        )

    final_decision = single_agent.temp_responses(final_prompt, img_path=img_path)

    trace.update({
        "final_prompt": final_prompt,
        "fewshot_builder_trace": medical_agent.export_trace(),
        "single_agent_trace": single_agent.export_trace(),
    })

    if return_trace:
        return final_decision, trace
    return final_decision

def process_intermediate_query(question, examplers, model, args, img_path=None, return_trace=False):
    trace = {
        "mode": "intermediate",
        "question": question,
        "img_path": img_path,
        "console_log": [],
        "recruitment": {},
        "hierarchy": {},
        "fewshot_examplers": [],
        "experts": [],
        "initial_opinions": [],
        "rounds": [],
        "interaction_table": None,
        "moderator": {},
    }

    def log(message=""):
        trace["console_log"].append(message)
        print(message)

    def log_info(message):
        trace["console_log"].append(message)
        cprint(message, 'yellow', attrs=['blink'])

    log_info("[INFO] Step 1. Expert Recruitment")
    recruit_prompt = concise_system(
        "You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."
    )

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-3.5')
    tmp_agent.chat(recruit_prompt)

    num_agents = 5
    recruitment_request = (
        f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. "
        "Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\n"
        "Also, you need to specify the communication structure between experts "
        "(e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\n"
        "For example, if you want to recruit five experts, you answer can be like:\n"
        "1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n"
        "2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n"
        "3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n"
        "4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n"
        "5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\n"
        "Please answer in above format, and do not include your reason."
    )
    recruitment_request = concise_user(
        recruitment_request,
        "Keep each expertise description short. Keep the whole answer compact."
    )
    recruited = tmp_agent.chat(recruitment_request)
    agents_data = parse_recruited_experts(recruited, num_agents)

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)
    hierarchy_agents, hierarchy_details = parse_hierarchy(agents_data, agent_emoji, return_details=True)

    trace["recruitment"] = {
        "system_prompt": recruit_prompt,
        "request_prompt": recruitment_request,
        "response": recruited,
    }
    trace["hierarchy"] = {
        "details": hierarchy_details,
    }

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_name, description = agent[0].split(' - ', 1)
        agent_role = agent_name.split('.')[1].strip().lower()
        description = description.strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    expert_records_by_role = {}
    for idx, agent in enumerate(agents_data):
        try:
            agent_name, description = agent[0].split(' - ', 1)
            agent_role = agent_name.split('.')[1].strip().lower()
            description = description.strip().lower()
        except:
            continue

        inst_prompt = concise_system(
            f"You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."
        )
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model)
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

        expert_record = {
            "index": idx + 1,
            "emoji": agent_emoji[idx],
            "name": agent_name.strip(),
            "role": agent_role,
            "description": description,
            "raw_hierarchy": agent[1],
            "hierarchy_detail": hierarchy_details[idx] if idx < len(hierarchy_details) else None,
            "system_prompt": inst_prompt,
        }
        trace["experts"].append(expert_record)
        expert_records_by_role[agent_role] = expert_record

    for idx, agent in enumerate(agents_data):
        try:
            agent_name, description = agent[0].split(' - ', 1)
            log(f"Agent {idx+1} ({agent_emoji[idx]} {agent_name.strip()}): {description.strip()}")
        except:
            log(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction=concise_system('You are a helpful medical agent.'), role='medical expert', model_info=model)
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(
                concise_user(
                    f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}",
                    "Use only one short sentence."
                )
            )

            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question
            trace["fewshot_examplers"].append({
                "question": exampler['question'],
                "options": exampler['options'],
                "answer_idx": exampler['answer_idx'],
                "answer": exampler['answer'],
                "generated_reason": exampler_reason,
            })

    log()
    log_info("[INFO] Step 2. Collaborative Decision Making")
    log_info("[INFO] Step 2.1. Hierarchy Selection")
    hierarchy_tree = render_tree(hierarchy_agents[0])
    trace["hierarchy"]["tree"] = hierarchy_tree
    if hierarchy_tree:
        log(hierarchy_tree)
    log()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {
        f'Round {round_num}': {
            f'Turn {turn_num}': {
                f'Agent {source_agent_num}': {
                    f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)
                } for source_agent_num in range(1, num_agents + 1)
            } for turn_num in range(1, num_turns + 1)
        } for round_num in range(1, num_rounds + 1)
    }

    log_info("[INFO] Step 2.2. Participatory Debate")

    round_opinions = {n: {} for n in range(1, num_rounds + 1)}
    round_answers = {f'Round {n}': None for n in range(1, num_rounds + 1)}
    for role, agent in agent_dict.items():
        initial_prompt = concise_user(
            f"Given the examplers, please return your answer to the medical query among the option provided.\n\n"
            f"{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ",
            SHORT_REASON_RULE
        )
        opinion = agent.chat(initial_prompt, img_path=img_path)
        round_opinions[1][role.lower()] = opinion
        trace["initial_opinions"].append({
            "role": role.lower(),
            "prompt": initial_prompt,
            "response": opinion,
        })
        if role.lower() in expert_records_by_role:
            expert_records_by_role[role.lower()]["initial_opinion"] = opinion

    final_answer = None
    for n in range(1, num_rounds + 1):
        log(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(
            instruction=concise_system("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts."),
            role="medical assistant",
            model_info=model,
        )
        agent_rs.chat(concise_system("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts."))

        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())
        summarizer_prompt = concise_user(
            f"Here are some reports from different medical domain experts.\n\n{assessment}\n\n"
            "You need to complete the following steps\n"
            "1. Take careful and comprehensive consideration of the following reports.\n"
            "2. Extract key knowledge from the following reports.\n"
            "3. Derive the comprehensive and summarized analysis based on the knowledge\n"
            "4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\n"
            "You should output in exactly the same format as: Key Knowledge:; Total Analysis:",
            "Keep both sections very short. No long discussion."
        )
        summary_report = agent_rs.chat(summarizer_prompt, img_path=img_path)
        round_trace = {
            "round": n,
            "assessment": assessment,
            "summarizer": {
                "system_prompt": agent_rs.instruction,
                "prompt": summarizer_prompt,
                "response": summary_report,
            },
            "turns": [],
            "final_answers": [],
        }

        num_yes = 0
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            log(f"|_{turn_name}")
            turn_trace = {
                "turn": turn_num + 1,
                "participation": [],
            }

            num_yes = 0
            for idx, agent in enumerate(medical_agents):
                all_comments = "".join(
                    f"{source} -> Agent {idx+1}: {targets[f'Agent {idx+1}']}\n"
                    for source, targets in interaction_log[round_name][turn_name].items()
                )
                participate_prompt = concise_user(
                    "Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\n"
                    f"Opinions:\n{assessment if n == 1 else all_comments}",
                    "Return only yes or no."
                )
                participate = agent.chat(participate_prompt, img_path=img_path)
                participation_entry = {
                    "source_agent_index": idx + 1,
                    "source_role": agent.role,
                    "participate_prompt": participate_prompt,
                    "participate_response": participate,
                    "messages": [],
                }

                if 'yes' in participate.lower().strip():
                    choose_prompt = concise_user(
                        f"Enter the number of the expert you want to talk to:\n{agent_list}\n"
                        "For example, if you want to talk with Agent 1. Pediatrician, return just 1. "
                        "If you want to talk with more than one expert, please return 1,2 and don't return the reasons.",
                        "Return only digits and commas."
                    )
                    chosen_expert = agent.chat(choose_prompt, img_path=img_path)

                    chosen_experts = []
                    for raw_target in chosen_expert.replace('.', ',').split(','):
                        raw_target = raw_target.strip()
                        if not raw_target.isdigit():
                            continue
                        target_index = int(raw_target)
                        if target_index < 1 or target_index > num_agents or target_index == idx + 1:
                            continue
                        if target_index not in chosen_experts:
                            chosen_experts.append(target_index)

                    participation_entry["choose_prompt"] = choose_prompt
                    participation_entry["choose_response"] = chosen_expert
                    participation_entry["chosen_experts"] = chosen_experts

                    if not chosen_experts:
                        warning = f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): no valid target parsed from `{chosen_expert}`"
                        log(warning)
                        participation_entry["warning"] = warning
                        turn_trace["participation"].append(participation_entry)
                        continue

                    for target_index in chosen_experts:
                        specific_prompt = concise_user(
                            f"Please remind your medical expertise and then leave your opinion to an expert you chose "
                            f"(Agent {target_index}. {medical_agents[target_index-1].role}). You should deliver your opinion once you are confident "
                            "enough and in a way to convince other expert with a short reason.",
                            "Use one short sentence."
                        )
                        specific_question = agent.chat(specific_prompt, img_path=img_path)
                        console_line = (
                            f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> "
                            f"Agent {target_index} ({agent_emoji[target_index-1]} {medical_agents[target_index-1].role}) : {specific_question}"
                        )
                        log(console_line)
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {target_index}'] = specific_question
                        participation_entry["messages"].append({
                            "target_agent_index": target_index,
                            "target_role": medical_agents[target_index-1].role,
                            "prompt": specific_prompt,
                            "response": specific_question,
                            "console_line": console_line,
                        })

                    num_yes += 1
                else:
                    console_line = f" Agent {idx+1} ({agent_emoji[idx]} {agent.role}): \U0001f910"
                    log(console_line)
                    participation_entry["console_line"] = console_line

                turn_trace["participation"].append(participation_entry)

            if num_yes == 0:
                turn_trace["stopped_early"] = True
                round_trace["turns"].append(turn_trace)
                break

            round_trace["turns"].append(turn_trace)

        if num_yes == 0:
            final_answer = dict(round_opinions[n])
            round_trace["stopped_without_new_final_answers"] = True
            round_trace["final_answer_snapshot"] = dict(final_answer)
            round_trace["summarizer"]["agent_trace"] = agent_rs.export_trace()
            trace["rounds"].append(round_trace)
            break

        tmp_final_answer = {}
        for idx, agent in enumerate(medical_agents):
            final_prompt = concise_user(
                "Now that you've interacted with other medical experts, remind your expertise and the comments from other experts "
                f"and make your final answer to the given question:\n{question}\nAnswer: ",
                SHORT_REASON_RULE
            )
            response = agent.chat(final_prompt, img_path=img_path)
            tmp_final_answer[agent.role] = response
            round_trace["final_answers"].append({
                "agent_index": idx + 1,
                "role": agent.role,
                "prompt": final_prompt,
                "response": response,
            })

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer
        if n < num_rounds:
            round_opinions[n + 1] = dict(tmp_final_answer)
        round_trace["final_answer_snapshot"] = dict(tmp_final_answer)
        round_trace["summarizer"]["agent_trace"] = agent_rs.export_trace()
        trace["rounds"].append(round_trace)

    log('\nInteraction Log')
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents) + 1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents) + 1):
            if i == j:
                row.append('')
            else:
                i2j = any(
                    interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                    for k in range(1, len(interaction_log) + 1)
                    for l in range(1, len(interaction_log['Round 1']) + 1)
                )
                j2i = any(
                    interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                    for k in range(1, len(interaction_log) + 1)
                    for l in range(1, len(interaction_log['Round 1']) + 1)
                )

                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                else:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents) + 1)])

    table_text = myTable.get_string()
    trace["interaction_table"] = table_text
    log(table_text)

    log_info("\n[INFO] Step 3. Final Decision")

    moderator = Agent(
        concise_system("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision."),
        "Moderator",
        model_info=model,
    )
    moderator.chat(concise_system('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.'))

    if getattr(args, 'task', None) == 'svdh_be':
        decision_prompt = (
            "Given each agent's final answer, please review each agent's opinion and make the final answer "
            "to the SvdH bone erosion scoring question by majority vote or best-supported consensus.\n"
            "Return the final answer as JSON in this exact schema: {\"score\":0,\"reason\":\"short visual reason\"}.\n\n"
            f"Agent opinions:\n{final_answer}\n\nQuestion: {question}"
        )
        decision_prompt = concise_user(decision_prompt, "Return only compact JSON. Keep the reason extremely short.")
    else:
        decision_prompt = (
            "Given each agent's final answer, please review each agent's opinion and make the final answer to the question "
            f"by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}"
        )
        decision_prompt = concise_user(decision_prompt, "Keep the answer very short.")

    _decision = moderator.temp_responses(decision_prompt, img_path=img_path)
    final_decision = {'majority': _decision}

    moderator_line = f"{'\U0001F468\u200D\u2696\uFE0F'} moderator's final decision (by majority vote): {_decision}"
    log(moderator_line)
    log()

    trace["recruitment"]["agent_trace"] = tmp_agent.export_trace()
    trace["fewshot_builder_trace"] = medical_agent.export_trace()
    trace["round_answers"] = round_answers
    trace["moderator"] = {
        "system_prompt": moderator.instruction,
        "decision_prompt": decision_prompt,
        "response": _decision,
        "agent_trace": moderator.export_trace(),
    }
    trace["expert_agent_traces"] = [agent.export_trace() for agent in medical_agents]

    if return_trace:
        return final_decision, trace
    return final_decision

def process_advanced_query(question, model, args, img_path=None):
    print("[STEP 1] Recruitment")
    group_instances = []

    recruit_prompt = concise_system(
        """You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""
    )

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gpt-4o-mini')
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited = tmp_agent.chat(
        concise_user(
            f"Question: {question}\n\nYou should organize {num_teams} MDTs with different specialties or purposes and each MDT should have {num_agents} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Patient History Team (PHT)\nMember 1: Psychiatrist or Psychologist (Lead) - Addresses any psychological impacts of the chronic disease and its treatments, including issues related to voice changes, self-esteem, and coping strategies.\nMember 2: Physical Therapist - Offers exercises and strategies to maintain physical health and potentially support vocal function recovery indirectly through overall well-being.\nMember 3: Vocational Therapist - Assists the patient in adapting to changes in voice, especially if their profession relies heavily on vocal communication, helping them find strategies to maintain their occupational roles.\n\nGroup 4 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.",
            "Keep each description short and keep the whole plan compact."
        ),
        img_path=img_path
    )

    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    group_strings = ["Group " + group for group in groups]
    
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print()

        group_instance = Group(res_gs['group_goal'], res_gs['members'], question, img_path=img_path)
        group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment = group_instance.interact(comm_type='internal')
            initial_assessments.append([group_instance.goal, init_assessment])

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"

    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment = group_instance.interact(comm_type='internal')
            assessments.append([group_instance.goal, assessment])
    
    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    
    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision = group_instance.interact(comm_type='internal')
            final_decisions.append([group_instance.goal, decision])
    
    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"

    # STEP 3. Final Decision
    decision_prompt = concise_system(
        """You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    )
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=model)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(
        concise_user(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}""", SHORT_REASON_RULE),
        img_path=img_path
    )

    return final_decision
