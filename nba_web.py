# %%
import nest_asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import re, ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import unicodedata
import inspect
import json
import os
from typing import Callable, Dict, Any, List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from nltk.corpus import stopwords
from nba_records_search import search_records
from nba_toolkit import (get_seasons,
                        get_player_info,
                        get_player_awards,
                        get_player_stats,
                        get_all_time_leaders,
                        get_league_leaders,
                        get_draft_history,
                        get_player_games,
                        get_high_low,
                        get_best_game,
                        count_games,
                        get_triple_doubles,
                        get_team_info,
                        get_franchise_leaders,
                        get_games,
                        get_game,
                        get_awards,
                        get_league_standings,
                        get_team_year_by_year_stats,
                        get_nba_champions,
                        get_team_roster,
                        get_playoffs)

# %%
nest_asyncio.apply()

# %%
functions =[get_seasons,
            get_player_info,
            get_player_awards,
            get_player_stats,
            get_all_time_leaders,
            get_league_leaders,
            get_draft_history,
            get_player_games,
            get_high_low,
            get_best_game,
            count_games,
            get_triple_doubles,
            get_team_info,
            get_franchise_leaders,
            get_games,
            get_game,
            get_awards,
            get_league_standings,
            get_team_year_by_year_stats,
            get_nba_champions,
            get_team_roster,
            get_playoffs]

# %%
class ToolRegistry:
    """Sistema para registrar y ejecutar herramientas de manera dinámica."""

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, func: Callable):
        """
        Registra una función como herramienta disponible y genera su schema JSON.
        """
        tool_name = func.__name__
        self.tools[tool_name] = func
        self.tool_schemas[tool_name] = self._generate_schema(func)

    def _generate_schema(self, func: Callable) -> Dict[str, Any]:
        """
        Genera un schema JSON de la función, usando tipos y docstring,
        y extrae la descripción de cada parámetro desde la sección Args.
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # Parsear descripciones de parámetros desde la sección Args
        param_descriptions = {}
        args_match = re.search(r"Args:(.*?)(?:Returns:|$)", doc, re.DOTALL)
        if args_match:
            args_text = args_match.group(1)
            # Buscar líneas con formato: nombre (tipo): descripción
            for line in args_text.splitlines():
                line = line.strip()
                if not line or ':' not in line:
                    continue
                name_type, desc = line.split(':', 1)
                name = name_type.split('(')[0].strip()
                param_descriptions[name] = desc.strip()

        # Construir el schema
        schema = {
            "name": func.__name__,
            "description": doc.strip().split('\n')[0] if doc else f"Herramienta: {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }

        for param_name, param in sig.parameters.items():
            # Tipo por defecto
            param_type = "string"

            # Intentar inferir tipo desde la anotación
            if param.annotation != inspect.Parameter.empty:
                ann = param.annotation
                if ann == int:
                    param_type = "integer"
                elif ann == float:
                    param_type = "number"
                elif ann == bool:
                    param_type = "boolean"
                elif ann == list:
                    param_type = "array"
                elif hasattr(ann, '__origin__') and ann.__origin__ == list:
                    param_type = "array"

            # Usar la descripción parseada si existe
            param_desc = param_descriptions.get(param_name, f"Parámetro {param_name}")

            schema["parameters"]["properties"][param_name] = {
                "type": param_type,
                "description": param_desc
            }

            # Si no tiene valor por defecto, es requerido
            if param.default == inspect.Parameter.empty:
                schema["parameters"]["required"].append(param_name)

        return schema

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Devuelve la lista de herramientas disponibles con sus schemas."""
        return list(self.tool_schemas.values())

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Ejecuta una herramienta con los parámetros dados.
        Devuelve el resultado o un mensaje de error.
        """
        if tool_name not in self.tools:
            return f"❌ Error: Herramienta '{tool_name}' no encontrada"

        try:
            result = self.tools[tool_name](**kwargs)
            return result
        except Exception as e:
            return f"❌ Error ejecutando '{tool_name}': {str(e)}"


# %%
registry = ToolRegistry()

for func in functions:
    registry.register_tool(func)

# %%
# --- Cargar modelo ---
# Modelo disponible en Hugging Face Hub
model_path = "joanmii/nba-agent-flan-t5"  # Tu modelo en Hugging Face
base_model = "google/flan-t5-large"

# Intentar cargar desde local primero, sino desde HF Hub
try:
    if os.path.exists("./nba-agent-large"):
        tokenizer = AutoTokenizer.from_pretrained("./nba-agent-large")
        model = AutoModelForSeq2SeqLM.from_pretrained("./nba-agent-large")
        print("✅ Modelo cargado desde carpeta local")
    else:
        raise FileNotFoundError("Carpeta local no encontrada")
except:
    print("📥 Descargando modelo desde Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print("✅ Modelo descargado desde Hugging Face Hub")

# %%
# --- Parsing helpers ---
def parse_conditions(cond_str: str):
    conds = {}
    pattern = r"['\"]?(\w+)['\"]?\s*:\s*(\d+)"
    matches = re.findall(pattern, cond_str)
    for key, value in matches:
        conds[key] = int(value)
    return conds

def parse_output(raw_str: str):
    if "|" not in raw_str:
        raise ValueError(f"No se pudo parsear: {raw_str}")

    tool, params_str = [x.strip() for x in raw_str.split("|", 1)]
    
    # Transformaciones de nombres de herramientas incorrectas
    tool_corrections = {
        "get_roster": "get_team_roster"
    }
    
    # Aplicar corrección si existe
    if tool in tool_corrections:
        tool = tool_corrections[tool]
    
    params = {}

    key_value_pattern = r"(\w+)\s*=\s*(.*?)(?=,\s*\w+\s*=|$)"
    for match in re.finditer(key_value_pattern, params_str):
        key = match.group(1)
        value = match.group(2).strip()

        if key in ["over_conditions", "under_conditions"]:
            params[key] = parse_conditions(value)
        else:
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value.strip("'\"")
    return {"tool": tool, "params": params}

def ejecutar_con_registry(output_modelo, registry):
    if isinstance(output_modelo, str):
        output_modelo = parse_output(output_modelo)

    tool = output_modelo["tool"]
    params = output_modelo["params"]

    if tool not in registry.tools:
        raise ValueError(f"Herramienta {tool} no encontrada en el registro")
    print(f"Ejecutando herramienta: {tool} con parámetros: {params}")

    return registry.tools[tool](**params)

def normalize_input(text: str) -> str:
    text = text.lower()
    text = text.replace("¿", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_and_execute(query, model, tokenizer, registry):
    query = normalize_input(query)
    inputs = tokenizer(query, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("➡️ Modelo predijo:", output_str)

    try:
        parsed = parse_output(output_str)
    except Exception as e:
        return {"error": "parse_failed", "raw": output_str, "exception": str(e)}, None

    try:
        result = ejecutar_con_registry(parsed, registry)
    except Exception as e:
        return {"error": f"Error ejecutando {parsed['tool']}", "exception": str(e)}, None

    return result, parsed



# %%
def df_to_html(df: pd.DataFrame) -> str:
        def make_clickable(val):
                # Primero verificar si es None, NaN o string vacío
                if pd.isna(val) or val is None or str(val).lower() in ['none', 'nan']:
                    return ''
                
                if isinstance(val, str) and (val.startswith("http://") or val.startswith("https://")):
                        return f'<a href="{val}" target="_blank" class="text-blue-600 underline">{val}</a>'
                return val

        return f"""
        <div style="border:1px solid #d1d5db; border-radius:8px; background:white; max-height:400px; overflow:auto;">
            <table class="min-w-full border-collapse text-sm">
                <thead class="bg-gray-100 sticky top-0 z-10">
                    <tr>
                        {''.join(f'<th class="px-3 py-2 border border-gray-300 text-left font-medium text-gray-700 whitespace-nowrap bg-gray-100">{col}</th>' for col in df.columns)}
                    </tr>
                </thead>
                <tbody class="divide-y divide-gray-200">
                    {''.join('<tr class="hover:bg-gray-50">' + ''.join(f'<td class="px-3 py-2 border border-gray-300 whitespace-nowrap">{make_clickable(val)}</td>' for val in row) + '</tr>' 
                                     for row in df.values)}
                </tbody>
            </table>
        </div>
        """

# %%

# 🔹 Definir app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/favicon.ico")
def favicon():
    return FileResponse("./static/favicon.ico")

app.mount("/static", StaticFiles(directory="./static"), name="static")

# 🔹 Clase Query
class Query(BaseModel):
    question: str

# 🔹 Endpoint /ask
@app.post("/ask")
def ask(query: Query):
    # Ejecutar la función predicha
    result, meta = predict_and_execute(query.question, model, tokenizer, registry)

    print("[DEBUG] Resultado de la función:", result)
    print("[DEBUG] Meta del modelo:", meta)

    response = {"text": "", "table": "", "images": [], "extra": ""}

    # Verificar si hay error o no hay datos
    if (isinstance(result, dict) and "error" in result) or result is None:
        response["text"] = "Nada que mostrar, pregunta otra cosa"
        response["images"] = ["/static/zzz.jpg"]
        return response

    # Manejo de tuple: (df, segundo_valor, lista_urls)
    if isinstance(result, tuple):
        df = result[0] if isinstance(result[0], pd.DataFrame) else None
        segundo = result[1] if len(result) > 1 else None
        urls = result[2] if len(result) > 2 and isinstance(result[2], list) else []
        
        # Verificar si el DataFrame está vacío
        if df is not None and df.empty:
            response["text"] = "Nada que mostrar, pregunta otra cosa"
            response["images"] = ["/static/zzz.jpg"]
            return response
            
    else:
        df = result if isinstance(result, pd.DataFrame) else None
        segundo = None
        urls = []
        
        # Verificar si es un DataFrame vacío
        if df is not None and df.empty:
            response["text"] = "Nada que mostrar, pregunta otra cosa"
            response["images"] = ["/static/zzz.jpg"]
            return response

    print("[DEBUG] DataFrame:", df)
    print("[DEBUG] Segundo valor:", segundo)
    print("[DEBUG] URLs:", urls)

    # Procesar DataFrame
    if df is not None:
        response["table"] = df_to_html(df)
    # Mostrar segundo valor si es entero o string
    if isinstance(segundo, (int, str)):
        segundo_str = str(segundo)
        # Verificar si es una URL y formatearla como enlace
        if isinstance(segundo, str) and (segundo_str.startswith("http://") or segundo_str.startswith("https://")):
            response["extra"] = f'<a href="{segundo_str}" target="_blank" class="text-blue-600 underline hover:text-blue-800">{segundo_str}</a>'
        else:
            response["extra"] = segundo_str
    # Procesar imágenes
    if urls:
        response["images"] = urls

    return response

# 🔹 Endpoint /ask-records
@app.post("/ask-records")
def ask_records(query: Query):
    """Endpoint específico para buscar récords de la NBA"""
    records = search_records(query.question)
    
    response = {"text": "", "table": "", "images": [], "extra": ""}
    
    if not records:
        response["text"] = "No se encontraron récords relacionados con tu consulta. Intenta con otra pregunta."
        response["images"] = ["/static/zzz.jpg"]
        return response
    
    # Crear tabla HTML con los récords encontrados
    records_html = """
    <div style="border:1px solid #d1d5db; border-radius:8px; background:white; max-height:500px; overflow:auto;">
        <table class="min-w-full border-collapse text-sm">
            <thead class="bg-amber-100 sticky top-0 z-10">
                <tr>
                    <th class="px-3 py-2 border border-gray-300 text-left font-medium text-gray-700 bg-amber-100">📂 Categoría</th>
                    <th class="px-3 py-2 border border-gray-300 text-left font-medium text-gray-700 bg-amber-100">🏆 Récord</th>
                    <th class="px-3 py-2 border border-gray-300 text-left font-medium text-gray-700 bg-amber-100">📊 Descripción</th>
                </tr>
            </thead>
            <tbody class="divide-y divide-gray-200">
    """
    
    for record in records:
        # Formatear el nombre de la sección para que sea más legible
        seccion_display = record.get('seccion', '').replace('_', ' ').title()
        records_html += f"""
                <tr class="hover:bg-amber-50">
                    <td class="px-3 py-2 border border-gray-300 text-xs text-gray-600">{seccion_display}</td>
                    <td class="px-3 py-2 border border-gray-300 font-medium text-gray-800">{record['categoria']}</td>
                    <td class="px-3 py-2 border border-gray-300 text-gray-700">{record['record']}</td>
                </tr>
        """
    
    records_html += """
            </tbody>
        </table>
    </div>
    """
    
    response["table"] = records_html
    response["text"] = f"🏆 Encontré {len(records)} récord(s) relacionado(s) con tu consulta:"
    
    return response


# %%
# 🔹 Frontend en /
@app.get("/", response_class=HTMLResponse)
def serve_page():
  return HTMLResponse("""
  <!DOCTYPE html>
  <html lang=\"es\">

  <head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <script src=\"https://cdn.tailwindcss.com\"></script>
    <title>NBA Assistant</title>
  </head>

  <body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-2xl shadow-lg w-full max-w-7xl">
    <img src="/static/nba-logo.jpg" alt="NBA Logo" style="display:block; margin:auto; max-width:120px; margin-bottom:20px;">
    <h1 class="text-2xl font-bold mb-2 text-center text-blue-600">🏀 NBA Assistant</h1>
    
    <!-- Botones de navegación -->
    <div class="flex gap-4 mb-6 justify-center">
        <button onclick="window.location.href='/'" 
            class="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 transition border-2 border-blue-700">
            🏀 Asistente NBA
        </button>
        <button onclick="window.location.href='/records'" 
            class="bg-amber-500 text-white px-6 py-2 rounded hover:bg-amber-600 transition">
            🏆 Récords NBA
        </button>
    </div>
    
    <!-- Botón de información -->
    <div class="flex justify-center mb-4">
        <button id="toggleInfo" onclick="toggleExamples()" 
            class="bg-blue-500 hover:bg-blue-600 text-white rounded-full w-10 h-10 flex items-center justify-center transition-colors duration-200 shadow-lg">
            <span class="text-lg font-bold">i</span>
        </button>
    </div>
    
    <!-- Ejemplos de consultas (inicialmente ocultos) -->
    <div id="examplesSection" class="bg-blue-50 p-4 rounded-lg mb-4 border border-blue-200 hidden">
        <h3 class="font-semibold text-blue-800 mb-2 text-center">💡 Ejemplos de consultas:</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-blue-700 text-center">
            <span>• "Estadísticas por temporada de LeBron James"</span>
            <span>• "Triples dobles de Russell Westbrook en playoffs 2017-18"</span>
            <span>• "Mejores anotadores de 2023"</span>
            <span>• "Game 7 entre Oklahoma y Indiana Pacers en las finales de este año"</span>
            <span>• "Premios MVP de Stephen Curry"</span>
            <span>• "Roster actual de Golden State"</span>
            <span>• "Draft de 2003"</span>
            <span>• "Playoffs de 2024"</span>
        </div>
    </div>
    
    <input id="question" type="text" placeholder="Escribe tu pregunta..."
         class="w-full p-3 border rounded mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500"/>
    <button onclick="ask()"
        class="w-full bg-blue-500 text-white py-2 rounded hover:bg-blue-600 transition">
      Preguntar
    </button>
    <div id="answer" class="mt-4 p-6 bg-gray-50 rounded min-h-[200px] text-gray-700 max-h-[600px] overflow-y-auto"></div>
    </div>

    <script>
    function toggleExamples() {
        const examplesSection = document.getElementById('examplesSection');
        const toggleButton = document.getElementById('toggleInfo');
        
        if (examplesSection.classList.contains('hidden')) {
            examplesSection.classList.remove('hidden');
            toggleButton.classList.remove('bg-blue-500', 'hover:bg-blue-600');
            toggleButton.classList.add('bg-blue-600', 'hover:bg-blue-700');
        } else {
            examplesSection.classList.add('hidden');
            toggleButton.classList.remove('bg-blue-600', 'hover:bg-blue-700');
            toggleButton.classList.add('bg-blue-500', 'hover:bg-blue-600');
        }
    }

    async function ask() {
        const question = document.getElementById('question').value;
        const responseDiv = document.getElementById('answer');

        const loadingMessages = [
            'Buscando jugadores...',
            'Buscando equipos...',
            'Comparando estadísticas...',
            'Buscando partidos...',
            'Comparando temporadas...'
        ];
        let msgIndex = 0;
        
        // Función para mostrar gif + mensaje
        const showLoadingMessage = () => {
            responseDiv.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <img src="/static/loading.gif" alt="Cargando..." style="max-width: 80px; margin-bottom: 15px; display: block; margin-left: auto; margin-right: auto;">
                    <p style="font-size: 16px; color: #666; margin: 0;">${loadingMessages[msgIndex]}</p>
                </div>
            `;
        };
        
        // Mostrar mensaje inicial
        showLoadingMessage();
        
        // Cambiar mensaje cada 1.2 segundos
        const loadingInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % loadingMessages.length;
            showLoadingMessage();
        }, 1200);

        try {
            const res = await fetch('http://localhost:8000/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question })
            });
            const data = await res.json();

            let imagesHtml = '';
            let linksHtml = '';
            
            if (data.images && Array.isArray(data.images) && data.images.length > 0) {
                // Separar imágenes de enlaces de NBA.com
                const actualImages = [];
                const nbaLinks = [];
                
                data.images.forEach(url => {
                    if (url.startsWith('https://www.nba.com')) {
                        nbaLinks.push(url);
                    } else {
                        actualImages.push(url);
                    }
                });
                
                // Generar HTML para imágenes reales
                if (actualImages.length > 0) {
                    imagesHtml = `<div class='flex flex-wrap gap-2 mb-4 justify-center'>` + 
                        actualImages.map(url => `<img src='${url}' alt='Imagen' style='max-width:150px; max-height:150px; object-fit:contain; border-radius:8px; border:1px solid #ddd;'>`).join('') +
                        `</div>`;
                }
                
                // Generar HTML para enlaces de NBA.com
                if (nbaLinks.length > 0) {
                    linksHtml = `<div class='mb-4 p-3 bg-orange-50 border-l-4 border-orange-500 rounded'>` +
                        nbaLinks.map(url => `<a href='${url}' target='_blank' class='block text-orange-600 underline hover:text-orange-800 mb-1'>${url}</a>`).join('') +
                        `</div>`;
                }
            }

            clearInterval(loadingInterval);
            responseDiv.innerHTML = `
                ${imagesHtml}
                ${linksHtml}
                ${data.extra ? `<div class='mb-4 p-3 bg-blue-50 border-l-4 border-blue-500 text-blue-800 rounded'>${data.extra}</div>` : ''}
                <p class='mb-2'>${data.text}</p>
                ${data.table || ''}
            `;
        } catch (err) {
            clearInterval(loadingInterval);
            responseDiv.textContent = 'Error al contactar con el servidor.';
            console.error(err);
        }
    }
    </script>

  </body>

  </html>
  """)

# 🔹 Frontend de Récords en /records
@app.get("/records", response_class=HTMLResponse)
def serve_records_page():
    return HTMLResponse("""
    <!DOCTYPE html>
    <html lang=\"es\">

    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <script src=\"https://cdn.tailwindcss.com\"></script>
        <title>Récords NBA - NBA Assistant</title>
    </head>

    <body class="bg-gradient-to-br from-amber-50 to-yellow-100 flex items-center justify-center min-h-screen">
        <div class="bg-white p-8 rounded-2xl shadow-lg w-full max-w-7xl border-2 border-amber-200">
            <img src="/static/nba-logo.jpg" alt="NBA Logo" style="display:block; margin:auto; max-width:120px; margin-bottom:20px;">
            <h1 class="text-2xl font-bold mb-2 text-center text-amber-600">🏆 Récords NBA</h1>
            <p class="text-center text-gray-600 mb-6">Busca récords históricos de la NBA</p>
            
            <!-- Botones de navegación -->
            <div class="flex gap-4 mb-6 justify-center">
                <button onclick="window.location.href='/'" 
                    class="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 transition">
                    🏀 Asistente NBA
                </button>
                <button onclick="window.location.href='/records'" 
                    class="bg-amber-500 text-white px-6 py-2 rounded hover:bg-amber-600 transition border-2 border-amber-700">
                    🏆 Récords NBA
                </button>
            </div>
            
            <!-- Botón de información -->
            <div class="flex justify-center mb-4">
                <button id="toggleInfoRecords" onclick="toggleExamplesRecords()" 
                    class="bg-amber-500 hover:bg-amber-600 text-white rounded-full w-10 h-10 flex items-center justify-center transition-colors duration-200 shadow-lg">
                    <span class="text-lg font-bold">i</span>
                </button>
            </div>
            
            <!-- Ejemplos de consultas (inicialmente ocultos) -->
            <div id="examplesSectionRecords" class="bg-amber-50 p-4 rounded-lg mb-4 border border-amber-200 hidden">
                <h3 class="font-semibold text-amber-800 mb-2 text-center">💡 Ejemplos de consultas:</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-amber-700 text-center">
                    <span>• "Más puntos en un partido"</span>
                    <span>• "Récords de Michael Jordan"</span>
                    <span>• "Récords de playoffs"</span>
                    <span>• "Más triples en una parte"</span>
                    <span>• "Jugador más joven MVP"</span>
                </div>
            </div>
            
            <input id="question" type="text" placeholder="Pregunta sobre récords de la NBA..."
                class="w-full p-3 border-2 border-amber-300 rounded mb-4 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-amber-500"/>
            <button onclick="askRecords()"
                class="w-full bg-amber-500 text-white py-2 rounded hover:bg-amber-600 transition font-medium">
                🔍 Buscar Récords
            </button>
            <div id="answer" class="mt-4 p-6 bg-gray-50 rounded min-h-[200px] text-gray-700 max-h-[600px] overflow-y-auto border border-gray-200"></div>
        </div>

        <script>
        function toggleExamplesRecords() {
            const examplesSection = document.getElementById('examplesSectionRecords');
            const toggleButton = document.getElementById('toggleInfoRecords');
            
            if (examplesSection.classList.contains('hidden')) {
                examplesSection.classList.remove('hidden');
                toggleButton.classList.remove('bg-amber-500', 'hover:bg-amber-600');
                toggleButton.classList.add('bg-amber-600', 'hover:bg-amber-700');
            } else {
                examplesSection.classList.add('hidden');
                toggleButton.classList.remove('bg-amber-600', 'hover:bg-amber-700');
                toggleButton.classList.add('bg-amber-500', 'hover:bg-amber-600');
            }
        }

        async function askRecords() {
            const question = document.getElementById('question').value;
            const responseDiv = document.getElementById('answer');

            const loadingMessages = [
                'Buscando récords históricos...',
                'Revisando temporada regular...',
                'Consultando playoffs...',
                'Buscando en finales...',
                'Comparando estadísticas...'
            ];
            let msgIndex = 0;
            
            // Función para mostrar gif + mensaje
            const showLoadingMessage = () => {
                responseDiv.innerHTML = `
                    <div style="text-align: center; padding: 20px;">
                        <img src="/static/loading.gif" alt="Cargando..." style="max-width: 80px; margin-bottom: 15px; display: block; margin-left: auto; margin-right: auto;">
                        <p style="font-size: 16px; color: #d97706; margin: 0;">${loadingMessages[msgIndex]}</p>
                    </div>
                `;
            };
            
            // Mostrar mensaje inicial
            showLoadingMessage();
            
            // Cambiar mensaje cada 1.5 segundos
            const loadingInterval = setInterval(() => {
                msgIndex = (msgIndex + 1) % loadingMessages.length;
                showLoadingMessage();
            }, 1500);

            try {
                const res = await fetch('http://localhost:8000/ask-records', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await res.json();

                let imagesHtml = '';
                
                if (data.images && Array.isArray(data.images) && data.images.length > 0) {
                    imagesHtml = `<div class='flex flex-wrap gap-2 mb-4 justify-center'>` + 
                        data.images.map(url => `<img src='${url}' alt='Imagen' style='max-width:150px; max-height:150px; object-fit:contain; border-radius:8px; border:1px solid #ddd;'>`).join('') +
                        `</div>`;
                }

                clearInterval(loadingInterval);
                responseDiv.innerHTML = `
                    ${imagesHtml}
                    ${data.extra ? `<div class='mb-4 p-3 bg-amber-50 border-l-4 border-amber-500 text-amber-800 rounded'>${data.extra}</div>` : ''}
                    <p class='mb-2 text-amber-800 font-medium'>${data.text}</p>
                    ${data.table || ''}
                `;
            } catch (err) {
                clearInterval(loadingInterval);
                responseDiv.innerHTML = `
                    <div class="text-center text-red-600 p-4">
                        <p>❌ Error al contactar con el servidor.</p>
                        <p class="text-sm text-gray-600 mt-2">Verifica que el servidor esté ejecutándose.</p>
                    </div>
                `;
                console.error(err);
            }
        }

        // Permitir buscar con Enter
        document.getElementById('question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                askRecords();
            }
        });
        </script>

    </body>

    </html>
    """)



## uvicorn nba_web:app --reload
## http://127.0.0.1:8000/