"""
Módulo de búsqueda de récords NBA.
Contiene toda la funcionalidad para buscar y filtrar récords de baloncesto NBA.
"""

import json
import re
import unicodedata
import nltk
from nltk.corpus import stopwords
from typing import List, Dict

# Descargar stopwords si no están disponibles
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')


def search_records(query: str) -> List[Dict[str, str]]:
    """
    Busca récords en el archivo nba_records.json basado en la consulta del usuario.
    Algoritmo mejorado con sinónimos, búsqueda flexible y mejor scoring.
    
    Args:
        query (str): La pregunta del usuario sobre récords
        
    Returns:
        List[Dict]: Lista de récords encontrados ordenados por relevancia
    """
    try:
        # Cargar el archivo JSON de récords
        with open('nba_records.json', 'r', encoding='utf-8') as f:
            records_data = json.load(f)
        
        # Normalizar la consulta (eliminar acentos, caracteres especiales)
        def normalize_text(text):
            text = unicodedata.normalize('NFD', text.lower())
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
            text = re.sub(r'[^\w\s]', ' ', text)
            return ' '.join(text.split())
        
        query_normalized = normalize_text(query)
        results = []
        
        # Diccionario de sinónimos y variaciones mejorado
        synonyms = {
            # Estadísticas principales
            'puntos': ['punto', 'puntos', 'anotacion', 'anotar', 'gol', 'canasta', 'enceste', 'acierto'],
            'rebotes': ['rebote', 'rebotes', 'recuperacion', 'recuperar'],
            'asistencias': ['asistencia', 'asistencias', 'pase', 'pases', 'assist', 'ayuda'],
            'robos': ['robo', 'robos', 'steal', 'steals', 'intercepcion', 'quitar'],
            'tapones': ['tapon', 'tapones', 'block', 'blocks', 'bloqueo', 'bloquear'],
            'triples': ['triple', 'triples', 'three', 'tiro de tres', '3pt', 'triplero'],
            
            # Tipos de tiros
            'tiros': ['tiro', 'tiros', 'field goal', 'fg', 'lanzamiento', 'lanzar'],
            'libres': ['libre', 'libres', 'free throw', 'ft', 'personal', 'falta'],
            
            # Contextos temporales
            'partido': ['partido', 'partidos', 'game', 'games', 'encuentro', 'enfrentamiento'],
            'temporada': ['temporada', 'season', 'ano', 'campana', 'regular'],
            'carrera': ['carrera', 'career', 'total', 'historico', 'toda'],
            'playoffs': ['playoff', 'playoffs', 'eliminatoria', 'eliminatorias', 'postemporada'],
            'finales': ['final', 'finales', 'finals', 'championship'],
            'finales de NBA ganadas': ['finales de nba ganadas', 'finales de nba', 'campeonatos ganados', 'anillos ganados', 'títulos ganados', 'titulos ganados', 'campeonatos de nba'],
            
            # Categorías especiales
            'rookie': ['rookie', 'novato', 'debutante', 'primer ano', 'principiante'],
            'equipo': ['equipo', 'team', 'conjunto', 'franquicia'],
            'franquicia': ['equipo', 'conjunto', 'team', 'equipos'],
            'mvp': ['mvp', 'mejor jugador', 'mas valioso', 'valuable player'],
            'edad': ['edad', 'joven', 'viejo', 'mayor', 'menor', 'anos', 'old', 'young'],
            'consecutivo': ['consecutivo', 'consecutivos', 'seguido', 'seguidos', 'fila', 'racha']
        }
        
        # Diccionario expandido de jugadores con variaciones (AMPLIADO)
        players = {
            # Leyendas históricas
            'wilt': ['wilt', 'chamberlain', 'wilt chamberlain', 'big dipper'],
            'jordan': ['jordan', 'michael', 'michael jordan', 'mj', 'air jordan', 'his airness'],
            'lebron': ['lebron', 'james', 'lebron james', 'king james', 'bron'],
            'kobe': ['kobe', 'bryant', 'kobe bryant', 'black mamba', 'mamba'],
            'magic': ['magic', 'johnson', 'magic johnson', 'earvin', 'earvin johnson'],
            'bird': ['bird', 'larry', 'larry bird', 'hick from french lick'],
            'kareem': ['kareem', 'abdul-jabbar', 'kareem abdul-jabbar', 'abdul', 'jabbar'],
            'shaq': ['shaq', 'shaquille', 'oneal', 'shaquille oneal', 'diesel', 'big aristotle'],
            
            # Jugadores modernos con récords
            'curry': ['curry', 'stephen', 'stephen curry', 'steph', 'chef curry', 'baby faced assassin'],
            'thompson': ['thompson', 'klay', 'klay thompson', 'klay thompson'],
            'harden': ['harden', 'james', 'james harden', 'beard', 'the beard'],
            'westbrook': ['russell', 'westbrook', 'russell westbrook', 'russ', 'brodie'],
            'lillard': ['lillard', 'damian', 'damian lillard', 'dame', 'dame time'],
            'durant': ['durant', 'kevin', 'kevin durant', 'kd', 'easy money sniper'],
            'giannis': ['giannis', 'antetokounmpo', 'giannis antetokounmpo', 'greek freak'],
            
            # Asistentes históricos
            'stockton': ['stockton', 'john', 'john stockton'],
            'nash': ['nash', 'steve', 'steve nash'],
            'paul': ['paul', 'chris', 'chris paul', 'cp3', 'point god'],
            'skiles': ['skiles', 'scott', 'scott skiles'],
            
            # Defensivos/Reboteadores
            'hakeem': ['hakeem', 'olajuwon', 'hakeem olajuwon', 'dream', 'the dream'],
            'robinson': ['robinson', 'david', 'david robinson', 'admiral', 'the admiral'],
            'duncan': ['duncan', 'tim', 'tim duncan', 'big fundamental'],
            'garnett': ['garnett', 'kevin', 'kevin garnett', 'kg', 'big ticket'],
            'thurmond': ['thurmond', 'nate', 'nate thurmond'],
            'smith': ['smith', 'elmore', 'elmore smith'],
            
            # Otros con récords específicos
            'robertson': ['robertson', 'oscar', 'oscar robertson', 'big o'],
            'jokic': ['jokic', 'nikola', 'nikola jokic', 'joker'],
            'iverson': ['iverson', 'allen', 'allen iverson', 'ai', 'the answer'],
            'carter': ['carter', 'vince', 'vince carter', 'half man half amazing', 'air canada'],
            'nowitzki': ['nowitzki', 'dirk', 'dirk nowitzki', 'german wunderkind'],
            'malone': ['malone', 'karl', 'karl malone', 'mailman', 'the mailman'],
            'barkley': ['barkley', 'charles', 'charles barkley', 'round mound of rebound'],
            'pippen': ['pippen', 'scottie', 'scottie pippen'],
            
            # Jugadores con récords menos conocidos pero importantes
            'lawson': ['lawson', 'ty', 'ty lawson'],
            'ellis': ['ellis', 'dale', 'dale ellis'],
            'hield': ['hield', 'buddy', 'buddy hield'],
            'kenon': ['kenon', 'larry', 'larry kenon'],
            'green': ['green', 'ac', 'a.c. green', 'ac green'],
            'wade': ['wade', 'dwyane', 'dwyane wade', 'flash', 'd wade'],
            'anthony': ['anthony', 'carmelo', 'carmelo anthony', 'melo'],
            
            # Jugadores históricos adicionales
            'russell_bill': ['bill russell', 'russell bill', 'bill', 'mr clutch'],
            'west': ['west', 'jerry', 'jerry west', 'logo', 'the logo'],
            'hayes': ['hayes', 'elvin', 'elvin hayes', 'big e'],
            'erving': ['erving', 'julius', 'julius erving', 'dr j', 'doctor j'],
            'frazier': ['frazier', 'walt', 'walt frazier', 'clyde'],
            
            # Jugadores más recientes con récords
            'lillard': ['lillard', 'damian', 'damian lillard', 'dame', 'logo lillard'],
            'young': ['young', 'trae', 'trae young', 'ice trae'],
            'doncic': ['doncic', 'luka', 'luka doncic', 'wonder boy'],
            'morant': ['morant', 'ja', 'ja morant'],
            'booker': ['booker', 'devin', 'devin booker', 'book'],
            'mitchell': ['mitchell', 'donovan', 'donovan mitchell', 'spida'],
            'tatum': ['tatum', 'jayson', 'jayson tatum'],
            'embiid': ['embiid', 'joel', 'joel embiid', 'the process'],
            
            # Jugadores internacionales con récords
            'gasol': ['gasol', 'pau', 'pau gasol'],
            'ginobili': ['ginobili', 'manu', 'manu ginobili'],
            'parker': ['parker', 'tony', 'tony parker'],
            'sabonis': ['sabonis', 'arvydas', 'arvydas sabonis'],
            
            # Especialistas en records específicos
            'miller': ['miller', 'reggie', 'reggie miller'],
            'allen': ['allen', 'ray', 'ray allen', 'jesus shuttlesworth'],
            'korver': ['korver', 'kyle', 'kyle korver'],
            'rondo': ['rondo', 'rajon', 'rajon rondo'],
            'howard': ['howard', 'dwight', 'dwight howard', 'superman']
        }
        
        # Lista de stopwords mejorada: NLTK + palabras específicas del contexto NBA
        nltk_stopwords = set(stopwords.words('spanish'))
        domain_specific_stopwords = {
            # Palabras específicas del contexto NBA y deportivo
            'record', 'records', 'récord', 'récords', 'nba', 'basketball', 'baloncesto',
            'estadística', 'estadisticas', 'stat', 'stats', 'dato', 'datos',
            'ver', 'mostrar', 'buscar', 'encontrar', 'dame', 'dime', 'cuál', 'cuáles',
            'quien', 'quién', 'como', 'cómo', 'donde', 'dónde', 'cuando', 'cuándo',
        }
        
        # Combinar ambas listas
        all_stopwords = nltk_stopwords.union(domain_specific_stopwords)
        
        # Función para filtrar stopwords de una lista de palabras
        def filter_stopwords(words):
            """Filtra las stopwords (NLTK + específicas del dominio) y palabras muy cortas de una lista de palabras"""
            return [word for word in words if len(word) > 2 and word.lower() not in all_stopwords]
        
        # Función para encontrar todas las variaciones de una palabra en el texto
        def find_matches(text, word_dict):
            """Encuentra coincidencias usando el diccionario de sinónimos"""
            matches = []
            text_words = text.split()
            
            for category, variations in word_dict.items():
                for variation in variations:
                    if variation in text:
                        matches.append((category, variation))
            return matches
        
        # Función para calcular relevancia mejorada con similitud avanzada
        def calculate_relevance(record, query_matches):
            """Calcula la relevancia basada principalmente en keywords con prioridad a coincidencias exactas"""
            search_text = normalize_text(f"{record['categoria']} {record['record']}")
            categoria_lower = record['categoria'].lower()
            record_lower = record['record'].lower()
            score = 0
            
            # Limpiar la consulta
            query_clean = query_normalized.replace('mas', 'más').replace('quien', '').replace('cual', '').replace('que', '').strip()
            query_words = query_clean.split()
            
            # NUEVO SISTEMA PRINCIPAL: KEYWORDS-FIRST APPROACH
            
            # Factor 0: COINCIDENCIA EXACTA TOTAL DE CATEGORÍA (peso máximo 1000)
            if query_clean == categoria_lower:
                return 1000  # MÁXIMA PRIORIDAD para coincidencia exacta total
            
            # Factor 1: KEYWORDS SYSTEM CON PUNTUACIÓN DIFERENCIADA - SISTEMA PRINCIPAL
            if 'keywords' in record:
                keywords = [kw.lower() for kw in record['keywords']]
                
                # Palabras comparativas de baja prioridad (menos puntos)
                low_priority_words = {
                    'más', 'mas', 'mayor', 'highest', 'most', 'largest',
                    'menos', 'menor', 'lowest', 'fewest', 'smallest',
                    'mejor', 'best', 'peor', 'worst',
                    'primer', 'primera', 'primero', 'first',
                    'último', 'ultima', 'last',
                    'temporada', 'regular', 'season',
                    'partido', 'game', 'encuentro',
                    'playoffs', 'playoff', 'eliminatorias', 'postemporada'
                }
                
                keyword_matches_high = 0  # Palabras importantes (puntos, nombres, etc.)
                keyword_matches_low = 0   # Palabras comparativas/contextuales
                exact_match_bonus = 0
                
                # Verificar coincidencias de keywords con puntuación diferenciada
                for query_word in query_words:
                    if query_word in keywords:
                        if query_word in low_priority_words:
                            keyword_matches_low += 1
                        else:
                            keyword_matches_high += 1
                        
                        # Bonus extra si la palabra aparece también en la categoría
                        if query_word in categoria_lower:
                            exact_match_bonus += 50
                
                if keyword_matches_high > 0 or keyword_matches_low > 0:
                    # Puntuación diferenciada por tipo de keyword
                    high_priority_score = keyword_matches_high * 150  # 150 puntos por palabra importante
                    low_priority_score = keyword_matches_low * 30    # 30 puntos por palabra comparativa
                    keyword_score = high_priority_score + low_priority_score
                    score += keyword_score + exact_match_bonus
                    
                    # Bonus por cobertura total de la consulta (ajustado)
                    total_matches = keyword_matches_high + keyword_matches_low
                    coverage = total_matches / len(query_words) if len(query_words) > 0 else 0
                    if coverage >= 0.9:  # 90% o más de cobertura
                        score += 200  # Bonus alto por cobertura casi total
                    elif coverage >= 0.7:  # 70% o más de cobertura
                        score += 100  # Bonus medio por buena cobertura
                    elif coverage >= 0.5:  # 50% o más de cobertura
                        score += 50   # Bonus bajo por cobertura básica
                    
                    # BONUS ESPECIAL: Si todas las palabras importantes están en keywords
                    important_query_words = [w for w in query_words if w not in low_priority_words]
                    if important_query_words and all(word in keywords for word in important_query_words):
                        score += 300  # Bonus máximo por coincidencia total de palabras importantes
            
            # Factor 2: COINCIDENCIA PARCIAL EN CATEGORÍA (peso máximo 200)
            if query_clean in categoria_lower and len(query_clean) >= len(categoria_lower) * 0.6:
                score += 200  # ALTA PRIORIDAD para coincidencia parcial significativa
            elif any(word in categoria_lower for word in query_words):
                matches_in_categoria = sum(1 for word in query_words if word in categoria_lower)
                score += matches_in_categoria * 80  # 80 puntos por cada palabra en categoría
            
            # Factor 3: BÚSQUEDA POR JUGADOR (peso 150)
            for player_category, variations in players.items():
                for variation in variations:
                    if variation in query_clean:
                        # Verificar si el jugador aparece en el récord
                        if any(v in record_lower for v in variations):
                            score += 150  # ALTA PRIORIDAD para jugadores
                            break
            
            # Factor 4: COINCIDENCIAS EN EL RÉCORD (peso máximo 50)
            matches_in_record = sum(1 for word in query_words if word in record_lower)
            score += matches_in_record * 25  # 25 puntos por cada palabra en el récord
            
            # Factor 5: COINCIDENCIAS DE SINÓNIMOS (peso 30)
            for query_category, query_variation in query_matches['synonyms']:
                if query_category in categoria_lower or query_variation in categoria_lower:
                    score += 30
                if query_category in record_lower or query_variation in record_lower:
                    score += 15
            
            # SISTEMA DE PENALIZACIONES POR KEYWORDS IMPORTANTES FALTANTES
            if 'keywords' in record:
                keywords = [kw.lower() for kw in record['keywords']]
                
                # Mismas palabras de baja prioridad que arriba
                low_priority_words = {
                    'más', 'mas', 'mayor', 'highest', 'most', 'largest',
                    'menos', 'menor', 'lowest', 'fewest', 'smallest',
                    'mejor', 'best', 'peor', 'worst',
                    'primer', 'primera', 'primero', 'first',
                    'último', 'ultima', 'last',
                    'temporada', 'regular', 'season',
                    'partido', 'game', 'encuentro',
                    'playoffs', 'playoff', 'eliminatorias', 'postemporada'
                }
                
                # Filtrar stopwords de la consulta
                query_words_filtered = filter_stopwords(query_words)
                
                # Solo penalizar por palabras REALMENTE IMPORTANTES que faltan
                missing_important_words = 0
                for word in query_words_filtered:
                    if len(word) >= 3 and word not in low_priority_words:  # Solo palabras importantes
                        # Si la palabra importante no está en keywords y no es un sinónimo conocido
                        if word not in keywords:
                            # Verificar si es un sinónimo conocido
                            is_synonym = False
                            for syn_category, syn_variations in synonyms.items():
                                if word in syn_variations and syn_category in keywords:
                                    is_synonym = True
                                    break
                            
                            # Si no es sinónimo, es una palabra faltante importante
                            if not is_synonym:
                                missing_important_words += 1
                
                # Aplicar penalizaciones progresivas (más severas para palabras importantes)
                if missing_important_words > 0:
                    # Penalización alta: 100 puntos por cada palabra IMPORTANTE faltante
                    base_penalty = missing_important_words * 100
                    
                    # Contar solo palabras importantes en la consulta para el porcentaje
                    important_words_in_query = [w for w in query_words_filtered if w not in low_priority_words]
                    
                    if len(important_words_in_query) > 0:
                        missing_ratio = missing_important_words / len(important_words_in_query)
                        
                        # Penalización extra si faltan muchas palabras importantes
                        if missing_ratio >= 0.7:  # 70% o más de palabras importantes faltan
                            base_penalty += 300  # Penalización muy severa
                        elif missing_ratio >= 0.5:  # 50% o más de palabras importantes faltan
                            base_penalty += 150  # Penalización severa
                    
                    score -= base_penalty
            
            return max(score, 0)  # No permitir scores negativos
        
        # Obtener matches de la consulta
        query_matches = {
            'synonyms': find_matches(query_normalized, synonyms),
            'players': find_matches(query_normalized, players)
        }
        
        def search_in_section(section_data, section_name):
            """Busca en una sección específica del JSON con lógica mejorada"""
            found_records = []
            
            if isinstance(section_data, dict):
                for subsection_name, subsection_data in section_data.items():
                    if isinstance(subsection_data, list):
                        for record in subsection_data:
                            if isinstance(record, dict) and 'categoria' in record and 'record' in record:
                                relevance = calculate_relevance(record, query_matches)
                                
                                # Solo incluir si tiene relevancia mínima
                                if relevance > 0:
                                    found_records.append({
                                        'seccion': section_name,
                                        'subseccion': subsection_name,
                                        'categoria': record['categoria'],
                                        'record': record['record'],
                                        'relevance': relevance
                                    })
                    else:
                        # Recursión para subsecciones
                        found_records.extend(search_in_section(subsection_data, f"{section_name} - {subsection_name}"))
            elif isinstance(section_data, list):
                for record in section_data:
                    if isinstance(record, dict) and 'categoria' in record and 'record' in record:
                        relevance = calculate_relevance(record, query_matches)
                        
                        if relevance > 0:
                            found_records.append({
                                'seccion': section_name,
                                'subseccion': '',
                                'categoria': record['categoria'],
                                'record': record['record'],
                                'relevance': relevance
                            })
            
            return found_records
        
        # Buscar en todas las secciones
        for section_name, section_data in records_data.items():
            results.extend(search_in_section(section_data, section_name))
        
        # NUEVO SISTEMA DE FILTRADO INTELIGENTE
        
        # 1. Detectar si la búsqueda es específica de un jugador
        target_player = None
        query_words_lower = [word.lower() for word in query_normalized.split()]
        
        for player_key, variations in players.items():
            for variation in variations:
                if variation in query_normalized:
                    target_player = player_key
                    break
            if target_player:
                break
        
        # 2. Determinar umbral dinámico basado en el tipo y especificidad de búsqueda
        query_words_filtered = filter_stopwords(query_normalized.split())
        
        if target_player:
            # Si es búsqueda de jugador específico: umbral bajo, mostrar TODOS sus records
            min_threshold = 50  # Umbral bajo para incluir todos los records del jugador
            player_specific_search = True
        else:
            # Detectar búsquedas muy específicas que requieren mayor precisión
            specific_keywords = ['cuarto', 'quarter', 'tiempo', 'prorroga', 'overtime', 'debut', 
                               'primer', 'ultimo', 'minutos', 'segundos', 'edad', 'años', 
                               'consecutivo', 'seguidos', 'fila', 'racha']
            
            is_very_specific = any(keyword in query_normalized for keyword in specific_keywords)
            has_multiple_criteria = len(query_words_filtered) >= 4  # 4+ palabras importantes
            
            if is_very_specific or has_multiple_criteria:
                # Para búsquedas muy específicas: umbral MUY alto para máxima precisión
                min_threshold = 200  # Umbral alto para búsquedas muy específicas
            else:
                # Para búsquedas generales: umbral moderado
                min_threshold = 100  # Umbral normal para búsquedas generales
                
            player_specific_search = False
        
        # 3. Aplicar filtrado por umbral
        filtered_results = []
        for record in results:
            relevance = record.get('relevance', 0)
            
            if player_specific_search and target_player:
                # Para búsquedas de jugador específico: verificar que el record es del jugador correcto
                record_text = record['record'].lower()
                player_variations = players.get(target_player, [])
                
                if any(variation in record_text for variation in player_variations):
                    # Es del jugador buscado, aplicar umbral bajo
                    if relevance >= min_threshold:
                        filtered_results.append(record)
                # Si no es del jugador, no incluir (filtrado estricto)
            else:
                # Para búsquedas generales: aplicar umbral normal
                if relevance >= min_threshold:
                    filtered_results.append(record)
        
        results = filtered_results
        
        # 4. NUEVO: Filtrado por especificidad contextual
        # Si la búsqueda incluye contexto específico, filtrar por ese contexto
        contextual_keywords = {
            'cuarto': ['cuarto', 'quarter'],
            'parte': ['parte', 'half'],
            'partido': ['partido', 'game'],
            'temporada': ['temporada', 'season'],
            'carrera': ['carrera', 'career'],
            'prorroga': ['prorroga', 'prórroga', 'overtime'],
            'debut': ['debut', 'first game'],
            'playoffs': ['playoffs', 'playoff'],
            'finales': ['finales', 'finals'],
            'equipo': ['equipo', 'team', 'franquicia', 'franchise'],
            'all_star': ['all-star', 'allstar', 'all star'],
            'series': ['series', 'serie'],
            'premios': ['premios', 'premio', 'awards', 'award', 'trofeo', 'trofeos']
        }
        
        # Detectar si la consulta incluye contexto específico
        detected_contexts = []
        for context, variations in contextual_keywords.items():
            if any(var in query_normalized for var in variations):
                detected_contexts.append(context)
        
        # Si se detecta contexto específico, filtrar resultados
        if detected_contexts and not player_specific_search:
            context_filtered_results = []
            
            for record in results:
                categoria_lower = record['categoria'].lower()
                record_lower = record['record'].lower()
                
                # Verificar si el récord coincide con alguno de los contextos detectados
                context_match = False
                for context in detected_contexts:
                    context_variations = contextual_keywords[context]
                    
                    # Buscar el contexto en la categoría o en las keywords del récord original
                    if any(var in categoria_lower for var in context_variations):
                        context_match = True
                        break
                    
                    # También buscar en el récord completo para mayor precisión
                    if any(var in record_lower for var in context_variations):
                        context_match = True
                        break
                
                # Solo incluir si coincide con el contexto específico
                if context_match:
                    context_filtered_results.append(record)
            
            # Aplicar el filtrado contextual
            results = context_filtered_results
            
            print(f"🎯 Filtrado contextual aplicado para: {detected_contexts}")
            print(f"   Resultados después del filtrado: {len(results)}")
        
        # Si no hay resultados con el algoritmo mejorado, hacer búsqueda básica mejorada
        if not results:
            # Búsqueda de respaldo con scoring básico
            query_words = query_normalized.split()
            meaningful_query_words = filter_stopwords(query_words)  # Usar la función de stopwords ya definida
            
            for section_name, section_data in records_data.items():
                if isinstance(section_data, dict):
                    for subsection_name, subsection_data in section_data.items():
                        if isinstance(subsection_data, list):
                            for record in subsection_data:
                                if isinstance(record, dict) and 'categoria' in record and 'record' in record:
                                    search_text = normalize_text(f"{record['categoria']} {record['record']}")
                                    
                                    # Calcular score básico
                                    basic_score = 0
                                    for word in meaningful_query_words:
                                        if word in search_text:
                                            basic_score += 1
                                    
                                    if basic_score > 0:
                                        results.append({
                                            'seccion': section_name,
                                            'subseccion': subsection_name,
                                            'categoria': record['categoria'],
                                            'record': record['record'],
                                            'relevance': basic_score
                                        })
        
        # Ordenamiento avanzado por relevancia con agrupación
        def advanced_sort_key(record):
            relevance = record.get('relevance', 0)
            
            # Bonus por presencia de números (más específico)
            if any(char.isdigit() for char in record['record']):
                relevance += 0.5
                
            # Bonus por longitud apropiada (ni muy corto ni muy largo)
            record_length = len(record['record'])
            if 20 <= record_length <= 100:
                relevance += 0.3
            
            # Penalty por récords muy genéricos
            if len(record['record'].split()) < 3:
                relevance -= 0.2
                
            return relevance
        
        # Aplicar ordenamiento avanzado
        results.sort(key=advanced_sort_key, reverse=True)
        
        # Eliminar duplicados manteniendo el de mayor relevancia
        seen = set()
        unique_results = []
        for record in results:
            record_key = (record['categoria'], record['record'])
            if record_key not in seen:
                seen.add(record_key)
                unique_results.append(record)
        
        # Simplificar - mostrar todos los resultados únicos ordenados por relevancia
        final_results = unique_results
        
        # Limpiar el campo de relevancia antes de devolver
        for record in final_results:
            record.pop('relevance', None)
        
        return final_results if final_results else []
        
    except Exception as e:
        print(f"Error buscando récords: {e}")
        return []