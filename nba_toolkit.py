# -----------------------------
# Librerías / imports
# -----------------------------

from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import (
    playercareerstats,
    leagueleaders,
    alltimeleadersgrids,
    drafthistory,
    commonplayerinfo,
    playergamelogs,
    PlayerAwards,
    FranchiseLeaders,
    BoxScoreTraditionalV2,
    LeagueGameLog,
    TeamDetails,
    leaguestandingsv3,
    TeamPlayerDashboard,
    TeamYearByYearStats)
from thefuzz import process
import pandas as pd
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import time
from IPython.display import display, HTML, Image
import re

# -----------------------------
# Variables globales / constantes
# -----------------------------

AWARD_MAP = {
    "MVP": "NBA Most Valuable Player",
    "FMVP": "NBA Finals Most Valuable Player",
    "MVPs": "Most Valuable Player",
    "ALL_NBA": "All-NBA",
    "ALL_DEFENSIVE": "All-Defensive Team",
    "ALL_ROOKIE": "All-Rookie Team",
    "ALL_STAR": "All-Star",
    "CHAMPION": "Champion",
    "POM": "Player of the Month",
    "POW": "Player of the Week",
    "ROY": "Rookie of the Year",
    "MIP": "Most Improved Player",
    "6MOY":  "Sixth Man of the Year",
    "DPOY": "Defensive Player of the Year",
    "INSEASON_MVP": "In-Season Tournament Most Valuable Player",
    "ALL_TOURNAMENT": "In-Season Tournament All-Tournament",
    "CLUTCH": "Clutch Player of the Year"
}

AWARD_IDS = {
    "MVP": 33,
    "Defensive Player of the Year": 39,
    "Rookie of the Year": 35,
    "Sixth Man of the Year": 40,
    "Most Improved Player": 36,
    "Coach of the Year": 34,
    "Finals MVP": 43,
    "All-Star MVP": 53,
    "All-NBA 1st Team": 44,
    "All-NBA 2nd Team": 45,
    "All-NBA 3rd Team": 46,
    "All-Rookie 1st Team": 47,
    "All-Rookie 2nd Team": 48,
    "All-Defensive 1st Team": 49,
    "All-Defensive 2nd Team": 50
}

AWARD_ALIASES = {
    "MVP": "MVP",
    "DPOY": "Defensive Player of the Year",
    "ROY": "Rookie of the Year",
    "6MOTY": "Sixth Man of the Year",
    "MIP": "Most Improved Player",
    "COTY": "Coach of the Year",
    "FMVP": "Finals MVP",
    "ASG-MVP": "All-Star MVP",
    "ALL-NBA-1": "All-NBA 1st Team",
    "ALL-NBA-2": "All-NBA 2nd Team",
    "ALL-NBA-3": "All-NBA 3rd Team",
    "ALL-ROOKIE-1": "All-Rookie 1st Team",
    "ALL-ROOKIE-2": "All-Rookie 2nd Team",
    "ALL-DEF-1": "All-Defensive 1st Team",
    "ALL-DEF-2": "All-Defensive 2nd Team"
}

AWARD_GROUPS = {
    "ALL-NBA": ["All-NBA 1st Team", "All-NBA 2nd Team", "All-NBA 3rd Team"],
    "ALL-ROOKIE": ["All-Rookie 1st Team", "All-Rookie 2nd Team"],
    "ALL-DEF": ["All-Defensive 1st Team", "All-Defensive 2nd Team"]
}

POS_EQUIVALENCES = {
    "G": ["G", "PG", "SG"],
    "PG": ["PG"],
    "SG": ["G", "SG"],
    "F": ["F", "SF", "PF"],
    "SF": ["SF"],
    "PF": ["PF"],
    "C": ["C"]
}

SERIES_MAP = {
    "FINALS": "Finals",
    "ECF": "Eastern Conference Finals",
    "WCF": "Western Conference Finals",
    "ECSF": "Eastern Conference Semifinals",
    "WCSF": "Western Conference Semifinals",
    "ECFR": "Eastern Conference First Round",
    "WCFR": "Western Conference First Round",

    "CF": "Conference Finals",
    "CSF": "Conference Semifinals",
    "FR": "First Round"
}

# -----------------------------
# Funciones auxiliares
# -----------------------------

def safe_find_player(name: str):
    """Busca un jugador por nombre, tolerando errores de escritura."""
    all_players = players.get_players()
    names = [p['full_name'] for p in all_players]
    
    # Coincidencia exacta
    matches = [p for p in all_players if p['full_name'].lower() == name.lower()]
    if matches:
        return matches[0]
    
    # Coincidencia aproximada
    best_match, score = process.extractOne(name, names)
    if score > 80:  # confianza mínima
        return [p for p in all_players if p['full_name'] == best_match][0]
    
    raise ValueError(f"No se encontró ningún jugador parecido a '{name}'")

def get_team_id(abbreviation: str) -> int:
    """
    Devuelve el ID y nombre de un equipo dado su abreviatura.

    Args:
        abbreviation: abreviatura del equipo, por ejemplo "LAL"

    Returns:
        dict: {"id": team_id, "full_name": team_name, "abbreviation": abbreviation}
    
    Raises:
        ValueError si no se encuentra el equipo
    """
    all_teams = teams.get_teams()
    abbreviation = abbreviation.upper()
    
    for team in all_teams:
        if team['abbreviation'].upper() == abbreviation:
            return team['id']
    
    raise ValueError(f"No se encontró ningún equipo con la abreviatura '{abbreviation}'")

def get_seasons(player_name: str) -> Dict[str, List[str]]:
    """
    Devuelve las temporadas disponibles de un jugador separadas por tipo:
    Regular Season, All-Star y Playoffs. Convierte los season_id (ej. 22003)
    al formato '2003-04'.
    
    Returns:
        dict con claves:
            - "Regular Season"
            - "All-Star"
            - "Playoffs"
    """
    player = safe_find_player(player_name)
    player_id = player["id"]

    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    df = info.available_seasons.get_data_frame()

    seasons = {"Regular Season": [], "All Star": [], "Playoffs": [], "PlayIn": [], "NBACup": []}

    for sid in df["SEASON_ID"]:
        prefix, year = int(str(sid)[0]), int(str(sid)[1:])
        season_str = f"{year}-{str(year+1)[-2:]}"
        
        if prefix == 2:
            seasons["Regular Season"].append(season_str)
        elif prefix == 3:
            seasons["All Star"].append(season_str)
        elif prefix == 4:
            seasons["Playoffs"].append(season_str)
        elif prefix == 5:
            seasons["PlayIn"].append(season_str)
        elif prefix == 6:
            seasons["NBACup"].append(season_str)

    return seasons

# -----------------------------
# Player Functions
# -----------------------------


def get_player_info(player: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Obtiene información básica y estadística de un jugador usando CommonPlayerInfo.

    Args:
        player: nombre del jugador

    Returns:
        tuple: (DataFrame con info del jugador, lista de dicts para LLM)
    """
    player_info = safe_find_player(player)
    player_id = player_info["id"]

    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)

    df = info.common_player_info.get_data_frame()
    df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce').dt.date

    df['HEIGHT'] = df['HEIGHT'].str.split('-').apply(lambda x: round(int(x[0])*30.48 + int(x[1])*2.54, 1) if isinstance(x, list) else None)
    df['HEIGHT'] = df['HEIGHT'].astype(str) + ' cm'

    df['WEIGHT'] = pd.to_numeric(df['WEIGHT'], errors='coerce').apply(lambda x: round(x*0.453592, 1) if pd.notna(x) else None)
    df['WEIGHT'] = df['WEIGHT'].astype(str) + ' kg'

    df['TEAM_NAME'] = df['TEAM_CITY'] + ' ' + df['TEAM_NAME']

    df = df.drop(columns=['DISPLAY_LAST_COMMA_FIRST','DISPLAY_FI_LAST', 'PLAYER_SLUG','LAST_AFFILIATION', 'FIRST_NAME','LAST_NAME',
                          'GAMES_PLAYED_CURRENT_SEASON_FLAG', 'PLAYERCODE', 'TEAM_CODE', 'TEAM_ID', 'PERSON_ID', 'TEAM_ABBREVIATION',
                          'TEAM_CITY', 'DLEAGUE_FLAG', 'NBA_FLAG', 'GAMES_PLAYED_FLAG' ])
    df = df.rename(columns={"DISPLAY_FIRST_LAST": "NAME","TEAM_NAME": "TEAM", 'ROSTERSTATUS': 'STATUS'})

    
    df = df[['NAME', 'BIRTHDATE', 'POSITION', 'TEAM', 'JERSEY', 'COUNTRY', 'HEIGHT', 'WEIGHT', 'FROM_YEAR', 'TO_YEAR', 'STATUS',
             'SCHOOL', 'SEASON_EXP', 'DRAFT_YEAR', 'DRAFT_ROUND', 'DRAFT_NUMBER', 'GREATEST_75_FLAG']]
    df = df.reset_index(drop=True)
    dict_list = df.to_dict(orient="records")

    try:
        display(Image(url=f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=100))
    except:
        pass

    return df.T, dict_list


def get_player_awards(
    player_name: str,
    award: list[str] | str | None = None,
    season: str | int | None = None
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los premios y reconocimientos de un jugador NBA, 
    con opción de filtrar por premio y por temporada.

    Args:
        player_name (str): Nombre completo del jugador (ej. "LeBron James").
        award (str | list[str] | None): Premio(s) o palabra(s) clave para filtrar. 
                                        Ej: "MVP", ["MVP", "All-NBA"], None (todos).
        season (str | int | None): Temporada a filtrar, formato "2015-16" o "2016". 
                                   Si None, devuelve todas las temporadas.

    Returns:
        tuple:
            - pd.DataFrame con columnas como PERSON_ID, FIRST_NAME, LAST_NAME, TEAM,
              DESCRIPTION, SEASON, MONTH, WEEK, CONFERENCE, TYPE, SUBTYPE1-3.
            - Lista de diccionarios con los mismos datos.
    """
    player = safe_find_player(player_name)
    player_id = player["id"]

    awards_endpoint = PlayerAwards(player_id=player_id)
    df = awards_endpoint.player_awards.get_data_frame().reset_index(drop=True)

    # --- Filtro por premio ---
    if award is not None:
        if isinstance(award, str):
            award = [award]
        filter_awards = [str(AWARD_MAP.get(a.upper(), a)) for a in award]
        pattern = "|".join(re.escape(fa) for fa in filter_awards)
        df = df[df["DESCRIPTION"].str.contains(pattern, case=False, na=False, regex=True)].reset_index(drop=True)

    # --- Filtro por temporada ---
    if season is not None:
        season = str(season)
        df = df[df["SEASON"].astype(str).str.contains(season, case=False, na=False)].reset_index(drop=True)

    df['FIRST_NAME'] = df['FIRST_NAME'] + ' ' + df['LAST_NAME']
    df['MONTH'] = pd.to_datetime(df['MONTH'], errors='coerce').dt.strftime('%m/%Y')
    df['WEEK'] = pd.to_datetime(df['WEEK'], errors='coerce').dt.strftime('%d/%m/%Y')

    df = df.drop(columns=['PERSON_ID', 'LAST_NAME', 'TYPE', 'SUBTYPE1', 'SUBTYPE2', 'SUBTYPE3'])

    df = df.rename(columns={"FIRST_NAME": "PLAYER_NAME","DESCRIPTION": "AWARD", 'ROSTERSTATUS': 'STATUS', 'ALL_NBA_TEAM_NUMBER': 'ALL_NBA_TEAM'})

    df.loc[df['CONFERENCE'].str.startswith('16', na=False), 'CONFERENCE'] = None

    df = df.replace(r'^\s*$', None, regex=True)
    df.dropna(axis=1, how="all", inplace=True)

    try:
        display(Image(url=f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=100))
    except:
        pass

    return df, df.to_dict(orient="records")


def get_player_stats(player: str,
                     season: str = None,
                     per_mode: str = "PerGame",
                     season_type: str = "Regular Season",
                     career: bool = False,
                     ranking: bool = False,
                     stats: list[str] = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve estadísticas de un jugador o rankings según los parámetros indicados.
    
    Args:
        player: nombre del jugador
        season: temporada en formato 'YYYY-YY' (ej. '2022-23'), None = toda la carrera
        per_mode: "Totals", "PerGame" o "Per36"
        season_type: "Regular Season", "Playoffs" o "All Star"
        career: si True, devuelve career totals en vez de season totals
        ranking: si True, devuelve los rankings de la temporada en lugar de stats (solo Regular Season o Playoffs)
        stats: lista de estadísticas a devolver (ej. ["PTS", "REB", "AST"]). Si None → todas
    
    Returns:
        (DataFrame, list[dict]) → 
        - DataFrame con las columnas seleccionadas o todas disponibles
        - Lista de dicts para enviar a un LLM
    """

    # 1. Buscar jugador
    player_info = safe_find_player(player)
    player_id = player_info["id"]

    # 2. Llamar endpoint
    career_endpoint = playercareerstats.PlayerCareerStats(
        player_id=player_id,
        per_mode36=per_mode
    )

    # 3. Elegir dataset según season_type y ranking
    if ranking:
        if season_type == "Regular Season":
            df = career_endpoint.season_rankings_regular_season.get_data_frame()
        elif season_type == "Playoffs":
            df = career_endpoint.season_rankings_post_season.get_data_frame()
        else:
            raise ValueError(f"No existe ranking para season_type={season_type}")
    else:
        if career:
            if season_type == "Regular Season":
                df = career_endpoint.career_totals_regular_season.get_data_frame()
            elif season_type == "Playoffs":
                df = career_endpoint.career_totals_post_season.get_data_frame()
            elif season_type == "All Star":
                df = career_endpoint.career_totals_all_star_season.get_data_frame()
            else:
                raise ValueError(f"season_type inválido: {season_type}")
        else:
            if season_type == "Regular Season":
                df = career_endpoint.season_totals_regular_season.get_data_frame()
            elif season_type == "Playoffs":
                df = career_endpoint.season_totals_post_season.get_data_frame()
            elif season_type == "All Star":
                df = career_endpoint.season_totals_all_star_season.get_data_frame()
            else:
                raise ValueError(f"season_type inválido: {season_type}")

    # 4. Filtrar por temporada si corresponde
    if season is not None and not career:
        df = df[df["SEASON_ID"] == season]
        if df.empty:
            raise ValueError(f"No hay datos para {player} en {season} ({season_type})")

    df = df.drop(columns=['PLAYER_ID', 'LEAGUE_ID', 'TEAM_ID'])

    if not career:
        df['TEAM_ABBREVIATION'] = df['TEAM_ABBREVIATION'].apply(lambda abbr: get_team_full_name(abbr))
        df = df.rename(columns={"SEASON_ID": "SEASON","TEAM_ABBREVIATION": "TEAM"})
        if not ranking:
            df['PLAYER_AGE'] = pd.to_numeric(df['PLAYER_AGE'], errors='coerce').astype('Int64')
            first_cols = ['SEASON', 'TEAM', 'PLAYER_AGE', 'GP', 'GS', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK']
            df = df.sort_values(by="SEASON", ascending=False).reset_index(drop=True)

    if ranking:
            df = df.drop(columns=['PLAYER_AGE', 'GP', 'GS'])
            df.rename(columns=lambda c: c.split("_")[-1] + "_RANK" if c.startswith("RANK_") else c, inplace=True)
            rank_cols = [c for c in df.columns if c.endswith("_RANK")]
            for col in rank_cols:
                s = df[col] if isinstance(df[col], pd.Series) else df[col].iloc[:, 0]
                df[col] = pd.to_numeric(s, errors='coerce').astype('Int64')

            first_cols = ['SEASON', 'TEAM',  'MIN_RANK', 'PTS_RANK', 'REB_RANK', 'AST_RANK', 'STL_RANK', 'BLK_RANK']            
            df = df.sort_values(by="SEASON", ascending=False).reset_index(drop=True)
           
    if career:
        first_cols = ['GP', 'GS', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK']

    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    if stats is not None:
        first_cols = [col for col in ["SEASON", "TEAM"] if col in df.columns]
        if ranking:
            first_cols = [col for col in ["SEASON", "TEAM"] if col in df.columns]
            stats = [f"{s}_RANK" for s in stats]
        stats_cols = [s for s in stats if s in df.columns]
        missing = set(stats) - set(stats_cols)
        if missing:
            raise ValueError(f"Columnas no encontradas en dataset: {missing}")
        df = df[first_cols + stats_cols]

    try:
        display(Image(url=f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=100))
    except:
        pass
    
    return df.reset_index(drop=True), df.to_dict(orient="records")


def get_all_time_leaders(stat: str = "PTS",
                         top: int = 10,
                         per_mode: str = "Totals",
                         season_type: str = "Regular Season",
                         player_name: str | None = None
                         ) -> tuple[pd.DataFrame, list[dict], pd.DataFrame | None]:
    """
    Devuelve los líderes históricos en una estadística específica.
    Opcionalmente busca un jugador en el ranking (aunque no esté en el top inicial).

    Args:
        stat: estadística a consultar. Valores posibles:
              ["PTS", "AST", "REB", "STL", "BLK", "FGM", "FGA", "FG_PCT",
               "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
               "OREB", "DREB", "TOV", "PF", "GP"]
        top: número de líderes a devolver (por defecto 10)
        per_mode: "Totals" o "PerGame"
        season_type: "Regular Season", "Playoffs" o "All Star"
        player_name: nombre (o parte del nombre) del jugador a buscar en el ranking, opcional para buscar en que posición del ranking está un determinado jugador

    Returns:
        (DataFrame, list[dict], DataFrame|None) →
        - DataFrame con los líderes solicitados
        - Lista de dicts para pasar al LLM
        - DataFrame con el jugador buscado (o None si no se especifica)
    """
    # Si buscamos jugador, ampliar top a 1000
    query_top = 1000 if player_name else top

    # Llamada al endpoint
    leaders = alltimeleadersgrids.AllTimeLeadersGrids(
        per_mode_simple=per_mode,
        season_type=season_type,
        topx=query_top
    )

    # Mapear clave del parámetro a dataset
    mapping = {
        "PTS": leaders.pts_leaders.get_data_frame(),
        "AST": leaders.ast_leaders.get_data_frame(),
        "REB": leaders.reb_leaders.get_data_frame(),
        "STL": leaders.stl_leaders.get_data_frame(),
        "BLK": leaders.blk_leaders.get_data_frame(),
        "FGM": leaders.fgm_leaders.get_data_frame(),
        "FGA": leaders.fga_leaders.get_data_frame(),
        "FG_PCT": leaders.fg_pct_leaders.get_data_frame(),
        "FG3M": leaders.fg3_m_leaders.get_data_frame(),
        "FG3A": leaders.fg3_a_leaders.get_data_frame(),
        "FG3_PCT": leaders.fg3_pct_leaders.get_data_frame(),
        "FTM": leaders.ftm_leaders.get_data_frame(),
        "FTA": leaders.fta_leaders.get_data_frame(),
        "FT_PCT": leaders.ft_pct_leaders.get_data_frame(),
        "OREB": leaders.oreb_leaders.get_data_frame(),
        "DREB": leaders.dreb_leaders.get_data_frame(),
        "TOV": leaders.tov_leaders.get_data_frame(),
        "PF": leaders.pf_leaders.get_data_frame(),
        "GP": leaders.g_p_leaders.get_data_frame(),
    }

    # Validar entrada
    if stat not in mapping:
        raise ValueError(f"Estadística no soportada: {stat}. Usa una de {list(mapping.keys())}")

    df = mapping[stat]

    if player_name:
        player = safe_find_player(player_name)
        player_name = player["full_name"]
        mask = df["PLAYER_NAME"].str.contains(player_name, case=False, na=False)
        df = df[mask].copy()
        if df.empty:
            print(f"No se encontró ningún jugador que coincida con '{player_name}'")

    df = df.drop(columns=['PLAYER_ID'])

    return df, df.to_dict(orient="records")


def get_league_leaders(stat: str = "PTS",
                       season: str = "2024-25",
                       season_type: str = "Regular Season",
                       top: int = 10,
                       per_mode: str = "PerGame",
                       rookies: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los líderes de la liga en una estadística determinada,
    mostrando solo columnas clave: RANK, PLAYER, TEAM, GP y la estadística solicitada.

    Args:
        stat: estadística a consultar ("PTS", "AST", "REB", etc.)
        season: temporada en formato 'YYYY-YY'
        season_type: "Regular Season", "Playoffs", "All Star"
        top: número de líderes a devolver
        per_mode: "Totals" o "PerGame"
        rookies: True → solo jugadores novatos, False → todos los jugadores

    Returns:
        (DataFrame, lista de dicts) → DataFrame para pantalla, lista de dicts para LLM
    """
    scope = "Rookies" if rookies else 'S'

    leaders = leagueleaders.LeagueLeaders(
        season=season,
        scope=scope,
        season_type_all_star=season_type,
        stat_category_abbreviation=stat,
        per_mode48=per_mode,
    )

    df = leaders.league_leaders.get_data_frame()
    df = df.head(top).reset_index(drop=True)
    df['TEAM'] = df['TEAM'].apply(lambda abbr: get_team_full_name(abbr))

    columns_to_keep = ["RANK", "PLAYER", "TEAM", stat, "GP"]
    df = df[columns_to_keep]

    dict_list = df.to_dict(orient="records")
    return df, dict_list


def get_draft_history(season: str = None,
                      team: str = None,
                      overall_pick: int = None,
                      round_num: int = None,
                      top: int = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve información del Draft de la NBA según filtros.

    Args:
        season: año del draft en formato 'YYYY' (ej. '2003')
        team: abreviatura del equipo que seleccionó (ej. 'CLE')
        overall_pick: número global de elección (ej. 1 = LeBron)
        round_num: número de ronda (ej. 1 o 2)
        top: limitar al top N picks (ej. 10 para top10 del draft)
    
    Returns:
        (DataFrame, lista de dicts) → DataFrame para mostrar, lista de dicts para LLM
    """
    team_id = get_team_id(team) if team else ""
    
    draft = drafthistory.DraftHistory(
        season_year_nullable=season if season else "",
        team_id_nullable=team_id,
        overall_pick_nullable=overall_pick if overall_pick else "",
        round_num_nullable=round_num if round_num else "",
        topx_nullable=top if top else "",
    )

    df = draft.draft_history.get_data_frame()

    df['TEAM'] = df['TEAM_ABBREVIATION'].apply(lambda abbr: get_team_full_name(abbr))
    cols = [
        "OVERALL_PICK",
        "PLAYER_NAME",
        "TEAM",
        "SEASON",
        "ROUND_NUMBER",
        "ROUND_PICK",
        "ORGANIZATION",
        "ORGANIZATION_TYPE",
    ]
    df = df[cols]

    # Limitar al top N si se especifica
    if top:
        df = df.head(top).reset_index(drop=True)

    # Convertir a lista de diccionarios (para LLM)
    dict_list = df.to_dict(orient="records")

    return df, dict_list


def get_player_games(player_name: str, season: str = None, season_type: str = None, last_x: int = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los partidos de un jugador como DataFrame y lista de diccionarios.

    Args:
        player_name (str): Nombre del jugador (ej. "LeBron James").
        season (str | None): Temporada en formato 'YYYY-YY' (ej. '2022-23').
                             Si None → devuelve todos los partidos disponibles.
        season_type (str | None): "Regular Season", "Playoffs", "All Star", "PlayIn" o None.
                                  Si None → Regular Season + Playoffs.
        last_x (int | None): Si se especifica, devuelve solo los últimos `x` partidos.

    Returns:
        (pd.DataFrame, list[dict]) → DataFrame con los partidos, lista de diccionarios.
    """
    # Buscar player_id
    player = safe_find_player(player_name)
    player_id = player["id"]

    if season:
        types = [season_type] if season_type else ["Regular Season", "Playoffs", "PlayIn"]
        all_logs = []
        for t in types:
            try:
                logs = playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id,
                    season_nullable=season,
                    season_type_nullable=t,
                )
                df_temp = logs.player_game_logs.get_data_frame()
                time.sleep(1)
                if not df_temp.empty:
                    df_temp["SEASON_TYPE"] = t
                    all_logs.append(df_temp)
            except Exception:
                continue

        df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    else:
        # Toda la carrera
        season_dict = get_seasons(player_name)
        if season_type == "Regular Season":
            season_list = season_dict.get("Regular Season", [])
            types = ["Regular Season"]
        elif season_type == "Playoffs":
            season_list = season_dict.get("Playoffs", [])
            types = ["Playoffs"]
        elif season_type == "All Star":
            season_list = season_dict.get("All Star", [])
            types = ["All Star"]
        elif season_type == "PlayIn":
            season_list = season_dict.get("PlayIn", [])
            types = ["PlayIn"]
        else:
            season_list = season_dict.get("Regular Season", []) + season_dict.get("Playoffs", []) + season_dict.get("PlayIn", [])
            types = ["Regular Season", "Playoffs", "PlayIn"]

        all_logs = []
        for s in season_list:
            for t in types:
                try:
                    logs = playergamelogs.PlayerGameLogs(
                        player_id_nullable=player_id,
                        season_nullable=s,
                        season_type_nullable=t,
                    )
                    df_temp = logs.player_game_logs.get_data_frame()
                    time.sleep(1)
                    if not df_temp.empty:
                        df_temp["SEASON_TYPE"] = t
                        all_logs.append(df_temp)
                except Exception:
                    continue

        df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()

    if not df.empty:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
        df['MIN'] = df['MIN_SEC']
        df['SEASON_YEAR'] = df['SEASON_YEAR'] + ' ' + df['SEASON_TYPE']
        df = df.drop_duplicates(subset=['GAME_ID'], keep='first')
        df = df.drop(columns=['NICKNAME', 'TEAM_ABBREVIATION', 'PFD', 'WNBA_FANTASY_PTS', 'AVAILABLE_FLAG', 'MIN_SEC', 'TEAM_COUNT', 'SEASON_TYPE'])
        df = df.rename(columns={"SEASON_YEAR": "SEASON","TEAM_NAME": "TEAM"})
        prioridad = ['SEASON', 'PLAYER_NAME', 'TEAM', 'GAME_DATE', 'MATCHUP', 'WL','MIN', "PTS", "REB", "AST", 'STL', 'BLK']
        resto = [c for c in df.columns if c not in prioridad and not c.endswith("_ID")]
        resto = [c for c in resto if c not in prioridad and not c.endswith("_RANK")]

        df = df[prioridad + resto]


    if last_x is not None and not df.empty:
        df = df.sort_values(by="GAME_DATE", ascending=False).head(last_x).reset_index(drop=True)
    else:
        df = df.sort_values(by="GAME_DATE", ascending=False).reset_index(drop=True)


    dict_list = df.to_dict(orient="records") if not df.empty else []

    try:
        display(Image(url=f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png", width=100))
    except:
        pass
    return df, dict_list


def get_high_low(
    player_name: str, 
    stat: str, 
    season: str = None, 
    season_type: str = None, 
    low: bool = False,
    top: int = 1
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los partidos con los valores más altos o más bajos en la estadística indicada.
    
    Args:
        player_name (str): Nombre del jugador.
        stat (str): Estadística a evaluar (ej. "PTS").
        season (str | None): Temporada en formato 'YYYY-YY' o None para toda la carrera.
        season_type (str | None): Tipo de temporada (Regular Season, Playoffs, etc.).
        low (bool): Si True devuelve los valores más bajos, si False los más altos.
        top (int): Número de partidos a devolver (por defecto 1).
    
    Returns:
        tuple: 
            - pd.DataFrame con las filas seleccionadas.
            - list[dict] con los mismos datos en formato diccionario.
    """
    df, dict = get_player_games(player_name, season=season, season_type=season_type)

    if df.empty or stat not in df.columns:
        return pd.DataFrame(), []

    cols = [stat, "SEASON_TYPE", "SEASON_YEAR", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP", "MIN"]

    # Ordenamos según la estadística
    df_sorted = df.sort_values(by=stat, ascending=low)

    # Nos quedamos con el top X
    result = df_sorted.head(top)[cols + [c for c in df.columns if c not in cols]].reset_index(drop=True)

    dict_list = result.to_dict(orient="records")

    return result, dict_list


def get_best_game(
    player_name: str, 
    season: str = None, 
    season_type: str = None, 
    worst: bool = False,
    top: int = 1
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los mejores o peores partidos de un jugador según NBA_FANTASY_PTS.
    Todas las columnas se devuelven, pero las más importantes van al principio.

    Args:
        player_name (str): Nombre del jugador.
        season (str | None): Temporada en formato 'YYYY-YY' o None para toda la carrera.
        season_type (str | None): Tipo de temporada (Regular Season, Playoffs, etc.).
        worst (bool): Si True devuelve los peores partidos; si False los mejores.
        top (int): Número de partidos a devolver (por defecto 1).

    Returns:
        tuple: (DataFrame con las filas seleccionadas, lista[dict] con los datos)
    """
    df, dict = get_player_games(player_name, season=season, season_type=season_type)
    
    if df.empty or "NBA_FANTASY_PTS" not in df.columns:
        return pd.DataFrame(), []

    # Columnas prioritarias
    priority_cols = [
        "NBA_FANTASY_PTS", "PTS", "REB", "AST", "STL", "BLK",
        "SEASON", "PLAYER_NAME", 
        "TEAM", "GAME_DATE", "MATCHUP", "MIN"
    ]
    
    # Mantener todas las columnas, pero poner primero las prioritarias
    remaining_cols = [c for c in df.columns if c not in priority_cols]
    cols = priority_cols + remaining_cols

    # Ordenar por NBA_FANTASY_PTS (desc si best, asc si worst)
    df_sorted = df.sort_values(by="NBA_FANTASY_PTS", ascending=worst)

    # Tomar top X
    result = df_sorted.head(top)[cols].reset_index(drop=True)

    dict_list = result.to_dict(orient="records")

    return result, dict_list


def count_games(
    player_name: str,
    over_conditions: Dict | None = None,
    under_conditions: Dict | None = None,
    season: str = None,
    season_type: str = None
) -> tuple[int, pd.DataFrame]:
    """
    Cuenta el número de partidos en los que un jugador cumple condiciones de estadísticas,
    tanto por encima como por debajo de ciertos valores, y devuelve el DataFrame filtrado.

    Args:
        player_name (str): Nombre del jugador.
        over_conditions (dict | None): Diccionario con stats que deben ser > valor, ej. {"PTS": 30}.
        under_conditions (dict | None): Diccionario con stats que deben ser < valor, ej. {"TOV": 3}.
        season (str | None): Temporada en formato 'YYYY-YY' o None para toda la carrera.
        season_type (str | None): Tipo de temporada (Regular Season, Playoffs, All Star, etc.).

    Returns:
        tuple:
            - int: Número de partidos que cumplen las condiciones.
            - pd.DataFrame: DataFrame con columnas reordenadas (stats primero, luego contexto).
    """
    df, _ = get_player_games(player_name, season=season, season_type=season_type)

    if df.empty:
        return 0, pd.DataFrame()

    mask = pd.Series([True] * len(df))

    # Condiciones "over"
    if over_conditions:
        for stat, threshold in over_conditions.items():
            if stat in df.columns:
                mask &= df[stat] >= threshold

    # Condiciones "under"
    if under_conditions:
        for stat, threshold in under_conditions.items():
            if stat in df.columns:
                mask &= df[stat] <= threshold

    filtered = df[mask].reset_index(drop=True)

    if filtered.empty:
        return 0, pd.DataFrame()

    # Reordenar columnas: primero stats pedidas, luego columnas de contexto
    stat_cols = list(over_conditions.keys() if over_conditions else []) + \
                list(under_conditions.keys() if under_conditions else [])

    base_cols = ["SEASON", "PLAYER_NAME", "TEAM",
                 "GAME_DATE", "MATCHUP", "MIN", "PTS", "AST", "REB", "BLK", "STL",
                 "FGM", "FGA", "FG3M", "FG3A", "TOV", "PF", "FG_PCT", "FG3_PCT", "FT_PCT"]

    # Evitar duplicados
    base_cols = [c for c in base_cols if c not in stat_cols]

    filtered = filtered[stat_cols + base_cols]

    return len(filtered), filtered


def get_triple_doubles(
    player_name: str,
    season: str = None,
    season_type: str = None,
    dd2: bool = False
) -> int:
    """
    Devuelve el número de triples-dobles o dobles-dobles logrados por un jugador.

    Args:
        player_name (str): Nombre del jugador.
        season (str | None): Temporada en formato 'YYYY-YY' o None para toda la carrera.
        season_type (str | None): "Regular Season", "Playoffs" o None (ambos).
        dd2 (bool): Si True, cuenta dobles-dobles en lugar de triples-dobles.

    Returns:
        int: Número de triples-dobles o dobles-dobles según `dd2`.
    """
    col = "DD2" if dd2 else "TD3"

    if season_type is None:
        df_rs, _ = get_player_games(player_name, season=season, season_type="Regular Season")
        df_po, _ = get_player_games(player_name, season=season, season_type="Playoffs")
        count_rs = df_rs[col].sum() if col in df_rs else 0
        count_po = df_po[col].sum() if col in df_po else 0
        return count_rs + count_po
    else:
        df, _ = get_player_games(player_name, season=season, season_type=season_type)
        return df[col].sum() if col in df else 0
    

# -----------------------------
# Team Functions
# -----------------------------


def get_team_info(team: str, flag: str = None) -> pd.DataFrame:
    """
    Obtiene información de un equipo de la NBA según el flag indicado.
    
    Args:
    - team: Abreviatura del equipo.
    - flag (str): Tipo de información a devolver. Valores posibles:
        "championships" -> TeamAwardsChampionships
        "conf"         -> TeamAwardsConf
        "div"          -> TeamAwardsDiv
        "background"   -> TeamBackground
        "history"      -> TeamHistory
        "hof"          -> TeamHof
        "retired"      -> TeamRetired
        "social"       -> TeamSocialSites
    
    Returns:
    - dict: Diccionario con la información solicitada.
    """
    if flag is None:
        all_teams = teams.get_teams()
        for t in all_teams:
            if t['abbreviation'].upper() == team:
                return t
    team_id = get_team_id(team)
    team = TeamDetails(team_id)

    mapping = {
        "championships": team.team_awards_championships,
        "conf": team.team_awards_conf,
        "div": team.team_awards_div,
        "background": team.team_background,
        "history": team.team_history,
        "hof": team.team_hof,
        "retired": team.team_retired,
        "social": team.team_social_sites,
    }
    
    if flag not in mapping:
        raise ValueError(f"Flag '{flag}' no reconocido. Usa uno de: {list(mapping.keys())}")
    
    df = mapping[flag].get_data_frame().reset_index(drop=True)
    df.dropna(axis=1, how="all", inplace=True)
    if flag == 'background' and not df.empty:
        df = df.drop(columns=['TEAM_ID'])
        df = df.T

    logo = f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
    display(Image(url=logo, width=80))

    if flag in ['hof', 'retired']:
        df = df.drop(columns='PLAYERID', errors="ignore")
    
    df.dropna(axis=1, how="all", inplace=True)
        
    return df, df.to_dict(orient="records")


def get_franchise_leaders(team: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los líderes históricos de una franquicia en varias estadísticas,
    excluyendo las columnas que terminan en '_ID'.

    Args:
        team (str): Abreviatura del equipo (ej. "LAL").

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con los líderes de la franquicia.
            - list[dict]: Lista de diccionarios con los mismos datos.
    """
    team_id = get_team_id(team)

    leaders_endpoint = FranchiseLeaders(team_id=team_id)
    df = leaders_endpoint.franchise_leaders.get_data_frame()

    df = df[[col for col in df.columns if not col.endswith("_ID")]]

    data = []
    for col in df.columns:
        if col.endswith('_PLAYER'):
            continue
        data.append({
            'STAT': col,
            'PLAYER': df[f'{col}_PLAYER'].iloc[0],
            'TOTAL': df[col].iloc[0]
        })

    df = pd.DataFrame(data)

    logo = f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg"
    display(Image(url=logo, width=80))

    return df.reset_index(drop=True), df.to_dict(orient="records")


def get_games(
    team1: str | None = None,
    team2: str = None,
    season: str = "2024-25",
    season_type: str | None = None,
    last_x: int = None,
    month: int = None,
    game_date: str = None,
    date_from: str = None,
    date_to: str = None,
    home_away: str | None = None,  # "home" = local, "away" = visitante
    result: str | None = None,      # "W" = victorias, "L" = derrotas
    logo: bool = True
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los partidos de un equipo, opcionalmente frente a otro equipo, o de toda la liga con múltiples filtros avanzados.

    La función permite filtrar por temporada, tipo de temporada (Regular Season, Playoffs, PlayIn),
    mes, fecha exacta, rango de fechas, local/visitante, resultado y cantidad de últimos partidos.
    Siempre retorna los partidos ordenados de más recientes a más antiguos.

    Args:
        team1 (str): Abreviatura del equipo principal (ej. "LAL").
        team2 (str, optional): Abreviatura del equipo contrario para filtrar enfrentamientos directos.
        season (str, optional): Temporada en formato 'YYYY-YY'. Default: "2024-25".
        season_type (str, optional): Tipo de temporada ("Regular Season", "Playoffs", "PlayIn"). Default: None = todos.
        last_x (int, optional): Limita la cantidad de partidos devueltos a los más recientes.
        month (int, optional): Filtra partidos por mes (1 = enero, 12 = diciembre).
        game_date (str, optional): Filtra por fecha exacta (formato "YYYY-MM-DD").
        date_from (str, optional): Filtra partidos desde esta fecha (inclusive).
        date_to (str, optional): Filtra partidos hasta esta fecha (inclusive).
        home_away (str, optional): Filtra por local/visitante. Valores: "home" o "away".
        result (str, optional): Filtra por resultado del partido. Valores: "W" = victoria, "L" = derrota.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con los partidos filtrados. Columnas incluyen:
              ["SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE",
              "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM",
              "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
              "PLUS_MINUS", "VIDEO_AVAILABLE"].
            - list[dict]: Lista de diccionarios cdel DataFrame.
    """
    season_types = [season_type] if season_type else ["Regular Season", "Playoffs", "PlayIn"]
    all_dfs = []

    if logo:
        if team1:
            team1_id= get_team_id(team1)
            logo1 = f"https://cdn.nba.com/logos/nba/{team1_id}/primary/L/logo.svg"

            if team2:
                team2_id= get_team_id(team2)
                logo2 = f"https://cdn.nba.com/logos/nba/{team2_id}/primary/L/logo.svg"
                display(HTML(f"""
                        <div style="display: flex; align-items: center; gap: 20px;">
                            <img src="{logo1}" width="80">
                            <img src="{logo2}" width="80">
                        </div>
                        """))
            else:
                display(Image(url=logo1, width=80))

    for stype in season_types:
        params = {"season": season, "season_type_all_star": stype}
        if date_from:
            params["DateFrom"] = date_from
        if date_to:
            params["DateTo"] = date_to

        log = LeagueGameLog(**params)
        df = log.league_game_log.get_data_frame()

        if df.empty:
            continue

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
        mask = pd.Series(True, index=df.index)

        if team1:
            mask &= df['MATCHUP'].str.contains(team1.upper())
            if team2:
                mask &= df['MATCHUP'].str.contains(team2.upper())
        if month:
            mask &= pd.to_datetime(df['GAME_DATE']).dt.month == month
        if game_date:
            mask &= df['GAME_DATE'] == pd.to_datetime(game_date).date()
        if home_away:
            if home_away.lower() == "home":
                mask &= df['MATCHUP'].str.contains("vs")
            elif home_away.lower() == "away":
                mask &= df['MATCHUP'].str.contains("@")
        if result:
            mask &= df['WL'] == result.upper()

        filtered = df[mask].reset_index(drop=True)
        all_dfs.append(filtered)

    if all_dfs:
        result_df = pd.concat(all_dfs)
    else:
        result_df = pd.DataFrame()

    if last_x and not result_df.empty:
        result_df = result_df.head(last_x*2).reset_index(drop=True)

    prefix_map = {
        "2": "Regular Season",
        "3": "All Star",
        "4": "Playoffs",
        "5": "PlayIn",
        "6": "NBACup"
    }
    df = result_df
    
    df['OT'] = ((df['MIN'] - 240) / 25).clip(lower=0).astype(int)

    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    df["SEASON"] = df["SEASON_ID"].str[0].map(prefix_map)

    year = df["SEASON_ID"].str[1:].astype(int)
    df["SEASON"] = year.astype(str) + "-" + (year + 1).astype(str).str[-2:] + '     ' + df['SEASON']

    def get_opponent(row):
        teams = [t.strip() for t in row['MATCHUP'].replace('vs.', ',').replace('@', ',').split(',')]
        return teams[1] if teams[0] == row['TEAM_ABBREVIATION'] else teams[0]

    df['OPPONENT_ABBR'] = df.apply(get_opponent, axis=1)

    df = df.rename(columns={"TEAM_NAME": "TEAM"})

    df = df.drop(columns=['OPPONENT_ABBR', 'SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'MIN', 'VIDEO_AVAILABLE'])

    first_cols = ['SEASON', 'MATCHUP', 'GAME_DATE', 'TEAM','WL','PTS']
    other_cols = [c for c in df.columns if c not in first_cols + ['GAME_ID']]
    df = df[first_cols + other_cols + ['GAME_ID']]

    df = df.sort_values(by=['GAME_DATE','GAME_ID'], ascending=False).reset_index(drop=True)


    mask = df.duplicated(subset='GAME_ID', keep='first')
    df.loc[mask, ['SEASON', 'GAME_DATE', 'MATCHUP']] = ''
        
    dict_list = df.to_dict(orient="records")
    return df, dict_list


def get_game_stats(game_id: str, boxscore: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve estadísticas de un partido según el flag boxscore.

    Args:
        game_id (str): ID del partido (ej. '0022300001').
        boxscore (bool): 
            - True → estadísticas por jugador (player_stats)
            - False → estadísticas totales por equipo (team_stats)

    Returns:
        tuple: (DataFrame, lista de dicts)
    """
    box = BoxScoreTraditionalV2(game_id=game_id)

    if boxscore:
        df = box.player_stats.get_data_frame()
        if "MIN" in df.columns:
            df = df[df["MIN"].notna()]
    else:
        df = box.team_stats.get_data_frame()
        
    if df.empty:
        return "NO DATA", []

    df = df.reset_index(drop=True)

    df['TEAM'] = df['TEAM_ABBREVIATION'].map(get_team_full_name)
    team_ids = df['TEAM_ID'].unique()

    team1_id, team2_id = team_ids[:2]

    logo1 = f"https://cdn.nba.com/logos/nba/{team1_id}/primary/L/logo.svg"
    logo2 = f"https://cdn.nba.com/logos/nba/{team2_id}/primary/L/logo.svg"

    display(HTML(f"""
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="{logo1}" width="80">
        <img src="{logo2}" width="80">
    </div>
    """))
    
    drop_cols = [col for col in ["GAME_ID", "TEAM_ID", "PLAYER_ID", "COMMENT", 'TEAM_ABBREVIATION','TEAM_CITY', 'TEAM_NAME', 'MIN', 'NICKNAME'] if col in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Reordenar columnas
    if boxscore:  
        # Para boxscore → PTS, AST, REB primero
        priority_cols = [col for col in ["TEAM", "PLAYER_NAME","START_POSITION", "PTS", "REB", "AST", "STL", "BLK"] if col in df.columns]
    else:  
        # Para team stats → solo PTS primero
        priority_cols = [col for col in ["TEAM","PTS"] if col in df.columns]

    other_cols = [col for col in df.columns if col not in priority_cols]
    df = df[priority_cols + other_cols]
    float_cols = df.select_dtypes(include='float').columns
    df[float_cols] = df[float_cols].astype(int)
    
    dict_list = df.to_dict(orient="records")

    return df, dict_list

def get_game(
    team1: str,
    team2: str = None,
    game: int | None = None,
    season: str = "2024-25",
    season_type: str = "Regular Season",
    boxscore: bool = False
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve las estadísticas de un partido especificando equipos y opcionalmente el número de partido.

    Args:
        team1 (str): Abreviatura del primer equipo (obligatorio, ej. 'LAL').
        team2 (str | None): Abreviatura del segundo equipo (ej. 'GSW').
        game (int | None): Índice del partido (1 = primer partido jugado en la temporada).
                           Si None o excede el número de partidos → se devuelve el último.
        season (str): Temporada 'YYYY-YY'.
        season_type (str): "Regular Season" o "Playoffs".
        boxscore (bool): True → stats por jugador, False → stats por equipo.

    Returns:
        tuple(pd.DataFrame, list[dict]): DataFrame y lista de diccionarios con las estadísticas.
    """
    if not team1:
        raise ValueError("Debe proporcionarse team1.")

    # Obtener partidos filtrados
    df_games, _ = get_games(team1, team2, season, season_type, logo = False)
    if df_games.empty:
        raise Exception("No se encontraron partidos con los filtros proporcionados.")

    # Selección del partido
    if game is not None and 1 <= game <= len(df_games):
        selected_game = df_games.iloc[-(game*2)]  # game=1 → primer partido
    else:
        selected_game = df_games.iloc[0]  # último partido

    game_id = selected_game["GAME_ID"]

    return get_game_stats(game_id, boxscore=boxscore)


def get_awards(year: int = None, award: str = None, pos: str = None, team: str = None, last_x: int = None, logo: int = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Obtiene premios históricos de la NBA desde ESPN, soportando filtros combinados.
    
    Args:
         year (int | list[int], optional): Año de los premios. Si None, devuelve todos los años.
                                           Si es lista/tupla de 2 años, devuelve los datos entre ambos inclusive.
        award (str, optional): Premio a consultar   "MVP": "MVP",
                                                    "DPOY": "Defensive Player of the Year",
                                                    "ROY": "Rookie of the Year",
                                                    "6MOTY": "Sixth Man of the Year",
                                                    "MIP": "Most Improved Player",
                                                    "COTY": "Coach of the Year",
                                                    "FMVP": "Finals MVP",
                                                    "ASG-MVP": "All-Star MVP",
                                                    "ALL-NBA-1": "All-NBA 1st Team",
                                                    "ALL-NBA-2": "All-NBA 2nd Team",
                                                    "ALL-NBA-3": "All-NBA 3rd Team",
                                                    "ALL-ROOKIE-1": "All-Rookie 1st Team",
                                                    "ALL-ROOKIE-2": "All-Rookie 2nd Team",
                                                    "ALL-DEF-1": "All-Defensive 1st Team",
                                                    "ALL-DEF-2": "All-Defensive 2nd Team",
                                                    "ALL-NBA": ["All-NBA 1st Team", "All-NBA 2nd Team", "All-NBA 3rd Team"],
                                                    "ALL-ROOKIE": ["All-Rookie 1st Team", "All-Rookie 2nd Team"],
                                                    "ALL-DEF": ["All-Defensive 1st Team", "All-Defensive 2nd Team"].
        pos (str, optional): Filtrar por posición (ej. 'PG', 'G', 'SF', 'F', 'C').
        team (str, optional): Filtrar por abreviatura del equipo (ej. 'LAL').
        last_x (int, optional): Si year es None, devuelve solo los últimos x ganadores por premio.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con los premios y jugadores.
            - list[dict]: Lista de diccionarios con los mismos datos.
    """
    results = []
    time.sleep(0.5)

    if award:
            award_lower = award.lower()
            award = next((full_name for alias, full_name in AWARD_ALIASES.items() if alias.lower() == award_lower), award)

        
    # Validación: si award es None, year solo puede ser un año simple
    if award is None and isinstance(year, (list, tuple)):
        year = max(year)


    if award is None and year:  # Scraping por año completo (todos los premios)
        url = f"https://www.espn.com/nba/history/awards/_/year/{year}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="tablehead")
        current_award = None

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                award_text = cells[0].get_text(strip=True)
                player_team = cells[1].get_text(strip=True)
                stats = cells[2].get_text(strip=True)

                if award_text != '':
                    current_award = award_text

                fg = ppg = rpg = apg = blk = None
                coach_stats = None
                player = player_team
                team_text = None

                if "," in player_team:
                    player, team_text = [x.strip() for x in player_team.rsplit(",", 1)]

                if current_award == "Coach of the Year":
                    coach_stats = stats
                else:
                    if stats != "No stats available.":
                        try:
                            stat_parts = [x.strip() for x in stats.split(",")]
                            stat_dict = {}
                            for part in stat_parts:
                                if ":" in part:
                                    key, val = part.split(":", 1)
                                    stat_dict[key.strip()] = float(val.strip())
                            fg = stat_dict.get("FG%", None)
                            ppg = stat_dict.get("PPG", None)
                            rpg = stat_dict.get("RPG", None)
                            apg = stat_dict.get("APG", None)
                            blk = stat_dict.get("BLKPG", None)
                        except:
                            fg = ppg = rpg = apg = blk = None

                if current_award != 'AWARD':
                    results.append({
                        "award": current_award,
                        "player": player,
                        "TEAM": team_text,
                        "PPG": ppg,
                        "RPG": rpg,
                        "APG": apg,
                        "BLKPG": blk,
                        "FG_PCT": fg,
                        "COACH_STATS": coach_stats
                    })
        time.sleep(0.5)

    elif award:  # Scraping por premio específico
        
        if award.upper() in AWARD_GROUPS:
            dfs = []
            all_results = []
            for sub_award in AWARD_GROUPS[award.upper()]:
                df_sub, results_sub = get_awards(year=year, award=sub_award, pos=pos, team=team, last_x=last_x, logo = 1)
                if not df_sub.empty:
                    df_sub.insert(0, "AWARD", sub_award)  # Agregar columna AWARD al DataFrame del sub-premio
                    for r in results_sub:
                        r["AWARD"] = sub_award  # Agregar AWARD a la lista de diccionarios
                dfs.append(df_sub)
                all_results.extend(results_sub)
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            df = df.sort_values(["YEAR", "AWARD"], ascending=[False, True]).reset_index(drop=True)
            if team:
                team_id = get_team_id(team)
                display(Image(url=f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg", width=80))

            return df, all_results
        
        if award not in AWARD_IDS:
            raise ValueError(f"Premio inválido. Debe estar en AWARD_IDS: {list(AWARD_IDS.keys())}")
        
        award_id = AWARD_IDS[award]
        url = f"https://www.espn.com/nba/history/awards/_/id/{award_id}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.find("table", class_="tablehead")

        current_year = None
        for row in table.find_all("tr"):
            cells = row.find_all("td")

            if award == "Coach of the Year" and len(cells) >= 7:
                year_text = cells[0].get_text(strip=True)
                if year_text == "YEAR":
                    continue
                if year_text != '':
                    current_year = year_text
                else:
                    year_text = current_year

                coach = cells[1].get_text(strip=True)
                team_text = cells[2].get_text(strip=True)
                wl = cells[3].get_text(strip=True)
                playoffs = cells[4].get_text(strip=True)
                career = cells[5].get_text(strip=True)
                exp = cells[6].get_text(strip=True)

                results.append({
                    "YEAR": int(year_text),
                    "COACH": coach,
                    "TEAM": team_text,
                    "W_L": wl,
                    "PLAYOFFS_W_L": playoffs,
                    "CAREER_W_L": career,
                    "EXP": exp
                })
            elif len(cells) >= 4:
                year_text = cells[0].get_text(strip=True)
                if year_text == "YEAR":
                    continue
                if year_text != '':
                    current_year = year_text
                else:
                    year_text = current_year

                player = cells[1].get_text(strip=True)
                pos_text = cells[2].get_text(strip=True)
                team_text = cells[3].get_text(strip=True)

                fg = ppg = rpg = apg = blk = None
                if len(cells) >= 9 and cells[4].get_text(strip=True) != "No stats available.":
                    fg = cells[4].get_text(strip=True)
                    ppg = cells[5].get_text(strip=True)
                    rpg = cells[6].get_text(strip=True)
                    apg = cells[7].get_text(strip=True)
                    blk = cells[8].get_text(strip=True)

                results.append({
                    "YEAR": int(year_text),
                    "PLAYER_NAME": player,
                    "POS": pos_text,
                    "TEAM": team_text,
                    "PPG": float(ppg) if ppg else None,
                    "RPG": float(rpg) if rpg else None,
                    "APG": float(apg) if apg else None,
                    "BLKPG": float(blk) if blk else None,
                    "FG_PCT": float(fg) if fg else None
                })

        time.sleep(0.5)

        # Filtrar por year o rango de years
        if year:
            if isinstance(year, (list, tuple)) and len(year) == 2:
                start, end = sorted(year)
                results = [r for r in results if start <= r["YEAR"] <= end]
            else:
                results = [r for r in results if r["YEAR"] == int(year)]

        # Filtros pos y team
        if pos:
            pos_upper = pos.upper()
            valid_positions = POS_EQUIVALENCES.get(pos_upper, [pos_upper])
            results = [r for r in results if r.get("POS") in valid_positions]

        if team:
            team_full_name = get_team_full_name(team)
            results = [r for r in results if r["TEAM"] == team_full_name]

        # Filtrar últimos X años
        if last_x is not None:
            last_years = sorted({r["YEAR"] for r in results}, reverse=True)[:last_x]
            results = [r for r in results if r["YEAR"] in last_years]

    df = pd.DataFrame(results)
    df.dropna(axis=1, how="all", inplace=True)

    

    if team:
        team_id = get_team_id(team)
        if not logo:
            display(Image(url=f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg", width=80))

    return df, results


def get_league_standings(season: str, conference: str | None = None, filter: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve la tabla de posiciones de la NBA para una temporada.

    Args:
        season (str): Temporada en formato 'YYYY-YY' (ej: '2024-25').
        conference (str|None): "East", "West" o None (default = ambos).
        filter (str|None): Filtra columnas por coincidencia parcial de nombre.
                           Si es "basic", muestra solo columnas básicas (Conference, TEAM_ABBREVIATION, Team, Record, PlayoffRank).
                           Si es "months", muestra solo columnas de meses (Jan, Feb, Mar, Apr, Oct, Nov, Dec).
                           Streak para mostrar las estadíticas de racha de los equipos.
                           Home y road para las estadísticas de local y visitante.
                           Vs para ver estadísticas contra conferencias y divisiones.
                           Score o Points para ver estadísticas de resultados y puntos a favor y en contra,
                           etc.

    Returns:
        (pd.DataFrame, list[dict]) → DataFrame con standings y lista de diccionarios.
    """
    # Obtener standings desde nba_api
    standings = leaguestandingsv3.LeagueStandingsV3(
        league_id="00",
        season=season,
        season_type="Regular Season",
    )

    time.sleep(0.5)

    df = standings.standings.get_data_frame()

    # Eliminar columnas innecesarias
    df.drop(columns=["LeagueID", "SeasonID", "TeamSlug"], inplace=True)

    df["Team"] = df["TeamCity"] + " " + df["TeamName"]
    df.drop(columns=["TeamCity", "TeamName", "TeamID"], inplace=True)
    
    # Filtrar por conferencia si corresponde
    if conference:
        conference = conference.capitalize() 
        if conference not in ["East", "West"]:
            raise ValueError("conference debe ser 'East', 'West' o None")
        df = df[df["Conference"] == conference]

    df.dropna(axis=1, how="all", inplace=True)

    df.sort_values(by=["WINS", "LOSSES"], ascending=[False, True], inplace=True)

    # Reordenar columnas → Conference, TeamID, TEAM_ABBREVIATION primero
    first_cols = ["Conference",  "PlayoffRank", "Team", "Record", "WinPCT"]
    other_cols = [c for c in df.columns if c not in first_cols]

    if filter:
        if filter.lower() == "months":
            months = ["Jan", "Feb", "Mar", "Apr", "Oct", "Nov", "Dec"]
            other_cols = [c for c in df.columns if c in months]
        elif filter.lower() == 'basic':
            other_cols = []
        else:
            # Filtra por coincidencia parcial de nombre
            other_cols = [c for c in df.columns if filter.lower() in c.lower()]

    df = df[first_cols + other_cols]

    dict_list = df.to_dict(orient="records") if not df.empty else []

    return df, dict_list


def get_team_year_by_year_stats(
    team: str,
    per_mode_simple: str = "PerGame",
    stats: bool = False,
    playoffs: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Obtiene estadísticas históricas de un equipo NBA temporada por temporada.

    Args:
        team (str): Abreviatura del equipo (ej. "LAL").
        per_mode_simple (str): Modo de estadísticas (default "PerGame").
        stats (bool): Si True devuelve las estadísticas del equipo.
        playoffs (bool): Si True devuelve los resultados en playoffs del equipo.

    Returns:
        (pd.DataFrame, list[dict]) → DataFrame con estadísticas y lista de diccionarios.

    """
    team_id = get_team_id(team)
    team_data = TeamYearByYearStats(
        team_id=team_id,
        league_id="00",
        per_mode_simple=per_mode_simple,
    )

    df = team_data.team_stats.get_data_frame()
    df["TEAM"] = df["TEAM_CITY"] + " " + df["TEAM_NAME"]
    df = df.drop(columns=["TEAM_ID", "TEAM_CITY", "TEAM_NAME", "CONF_COUNT", "DIV_COUNT"], errors="ignore")

    # Ordenar columnas
    cols = df.columns.tolist()
    cols = ['TEAM'] + [col for col in cols if col != 'TEAM']
    df = df[cols]

    df = df.sort_values("YEAR", ascending=False).reset_index(drop=True)

    # Filtrado avanzado
    first_cols = ["YEAR", "TEAM"]
    stats_cols = ['PTS','FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'PF',
       'STL', 'TOV', 'BLK','PTS_RANK']
    other_cols = [c for c in df.columns if c not in first_cols and c not in stats_cols]

    if stats:
        df = df[first_cols + stats_cols]
    elif playoffs:
        po_cols = ['PO_WINS', 'PO_LOSSES', 'NBA_FINALS_APPEARANCE']
        df = df[first_cols + po_cols]
    else:
        df = df[first_cols + other_cols]

    # Convertir a lista de diccionarios
    dict_list = df.to_dict(orient="records") if not df.empty else []

    display(Image(url=f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg", width=80))

    return df, dict_list


def get_nba_champions(year: int = None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve los campeones de la NBA y los principales jugadores de playoffs.
    
    Args:
        year (int | None): año específico para filtrar (ej. 2025). Si None → todos los años.
    
    Returns:
        (pd.DataFrame, list[dict]): DataFrame con columnas year, champion, runnerup, mvp_finals,
                                     pts_leader_name, trb_leader_name, ast_leader_name, ws_leader_name,
                                     y lista de diccionarios para uso con LLM.
    """
    url = "https://www.basketball-reference.com/playoffs/"
    html = urlopen(url)
    soup = BeautifulSoup(html, "lxml")
    
    # Buscar tabla
    container = soup.find("div", {"id": "div_champions_index"})
    table = container.find("table") if container else None
    
    if not table:
        raise Exception("No se encontró la tabla de campeones")
    
    # Columnas a mantener
    keep_headers = ["year_id", "champion", "runnerup", "mvp_finals",
                    "pts_leader_name", "trb_leader_name", "ast_leader_name", "ws_leader_name"]
    
    # Filas
    rows = table.find("tbody").find_all("tr")
    data = []
    for row in rows:
        row_data = []
        for stat in keep_headers:
            cell = row.find(attrs={"data-stat": stat})
            text = cell.get_text(strip=True) if cell else None
            row_data.append(text)
        data.append(row_data)
    
    # DataFrame
    df = pd.DataFrame(data, columns=keep_headers)
    df = df.rename(columns={"year_id": "YEAR", 'runneru':'RUNNERUP', "pts_leader_name": "PTS_LEADER", 
                            'mvp_finals':'FINALS_MVP', "trb_leader_name": "REB_LEADER", 
                            "trb_leader_name": "REB_LEADER", "ast_leader_name": "AST_LEADER", "ws_leader_name": "WIN_SHARE_LEADER"})
    
    # Filtrar por año si se proporciona
    if year:
        df = df[df["YEAR"] == str(year)]
        
    dict_list = df.to_dict(orient="records")
    
    return df.reset_index(drop=True), dict_list


def get_team_roster(
    team: str,
    season: str,
    season_type: str = "Regular Season",
    per_mode: str = "PerGame",
    filter: str | None = None  # None | "rank" | "stats" | "keywords separados por espacio"
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Devuelve el roster de un equipo para una temporada específica usando TeamPlayerDashboard,
    ordenado por puntos (PTS) descendente.

    Args:
        team (str): Abreviatura del equipo en NBA, por ejemplo 'BOS' para Boston Celtics.
        season (str): Temporada en formato '2024-25'.
        season_type (str): Tipo de temporada, 'Regular Season' o 'Playoffs'. Default es 'Regular Season'.
        per_mode (str): "PerGame" o "Totals". Default = "PerGame".
        filter (str | None): 
            None -> todas las columnas,
            "rank" -> solo columnas con 'RANK',
            "stats" -> solo columnas sin 'RANK',
            "col1 col2 ..." -> muestra solo esas columnas.

    Returns:
        (pd.DataFrame, list[dict]): DataFrame con información de los jugadores, y lista de diccionarios.
    """
    # Obtener el ID del equipo desde la abreviatura
    team_id = get_team_id(team)

    # Llamada al endpoint
    dashboard = TeamPlayerDashboard(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
        per_mode_detailed=per_mode,
        get_request=True
    )

    # Obtener dataframe de jugadores y limpiar
    df_players = dashboard.players_season_totals.get_data_frame()
    if "GROUP_SET" in df_players.columns:
        df_players = df_players.drop(columns=["GROUP_SET"])

    # Ordenar por puntos
    df_players = df_players.sort_values(by="PTS", ascending=False).reset_index(drop=True)

    # Limpiar columnas y renombrar
    df_players = df_players.drop(
        columns=["PLAYER_ID", "NICKNAME", "WNBA_FANTASY_PTS_RANK", "WNBA_FANTASY_PTS"],
        errors="ignore"
    )
    df_players = df_players.rename(columns={"PLAYER_NAME": "PLAYER"})
    
    df_players["MIN"] = df_players["MIN"].round().astype(int)

    # Columnas base
    player_col = ["PLAYER"]
    base_cols = ["GP", "W", "L", "W_PCT"]


    if filter and filter.lower() not in ["rank", "stats"]:
        keywords = filter.split()
        selected_cols = []
        first_cols = []
        for kw in keywords:
            matches = [c for c in df_players.columns if kw.lower() in c.lower()]
            selected_cols.extend(matches)
            first_cols.append(kw.upper())
            
        selected_cols = list(set(selected_cols) - set(first_cols))
        final_cols = player_col + first_cols + selected_cols
        df_players = df_players.sort_values(by=final_cols[1], ascending=False).reset_index(drop=True)

    else:
        priority_cols = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]
        if filter is None:
            selected_cols = [c for c in df_players.columns if c not in player_col]
        elif filter.lower() == "rank":
            selected_cols = [c for c in df_players.columns if "RANK" in c.upper()]
            priority_cols = [f"{s}_RANK" for s in ["MIN", "PTS", "REB", "AST", "STL", "BLK"] if f"{s}_RANK" in df_players.columns]
            if "PTS_RANK" in df_players.columns:
                df_players = df_players.sort_values("PTS_RANK").reset_index(drop=True)
        elif filter.lower() == "stats":
            selected_cols = [c for c in df_players.columns if "RANK" not in c.upper()]
            if "PLAYER" in selected_cols:
                selected_cols.remove("PLAYER")
            if "PTS" in df_players.columns:
                df_players = df_players.sort_values("PTS", ascending=False).reset_index(drop=True)
        else:
            selected_cols = [c for c in df_players.columns if c not in player_col]

        # Quitar duplicados de priority y base_cols
        selected_cols = [c for c in selected_cols if c not in priority_cols + base_cols]

        final_cols = player_col + base_cols + priority_cols + selected_cols

    # Reordenar columnas finales
    df_players = df_players[[c for c in final_cols if c in df_players.columns]]
    
    if filter and "_rank" in final_cols[1].lower():
        df_players = df_players.iloc[::-1].reset_index(drop=True)

    dict_list = df_players.to_dict(orient="records")

    display(Image(url=f"https://cdn.nba.com/logos/nba/{team_id}/primary/L/logo.svg", width=80))

    return df_players, dict_list


def get_playoffs(year: int, series: str | None = None, games: bool = False) -> pd.DataFrame:
    """
    Obtiene información de los Playoffs NBA de un año desde Basketball Reference.

    Args:
        year (int):
            Año de los playoffs (ejemplo: 2020).

        series (str | None, opcional):
            Serie a filtrar. Se puede usar el nombre completo o una abreviatura:
            
                - "Finals" → Finals
                - "ECF" → Eastern Conference Finals
                - "WCF" → Western Conference Finals
                - "ECSF" → Eastern Conference Semifinals
                - "WCSF" → Western Conference Semifinals
                - "ECFR" → Eastern Conference First Round
                - "WCFR" → Western Conference First Round
                - "CF" → Conference Finals (ECF + WCF)
                - "CSF" → Conference Semifinals (ECSF + WCSF)
                - "FR" → First Round (ECFR + WCFR)
                - "East" → Todas las series del este
                - "West" → Todas las series del oeste                

            La búsqueda no distingue mayúsculas/minúsculas.  
            Si None, devuelve todas las series.

        games (bool, opcional):
            - False → devuelve un DataFrame con resultados de series.
            - True → devuelve un DataFrame con resultados de partidos.

    Returns:
        tuple[pd.DataFrame, list[dict]]:
            - DataFrame con la información solicitada.
            - Lista de diccionarios (orient="records") con los mismos datos.

    Ejemplos:
        >>> get_playoffs(2020, series="Finals")
        # Series de las Finales NBA 2020

        >>> get_playoffs(2019, series="CF", games=True)
        # Partidos de las Conference Finals 2019
    """
    if series:
        if isinstance(series, str):
            series = [series]

        normalized_series = []
        for s in series:
            key = s.upper() 
            if key in SERIES_MAP:
                normalized_series.append(SERIES_MAP[key])
            else:
                normalized_series.append(s)
        series = normalized_series

    # --- Scraping ---
    url = f"https://www.basketball-reference.com/playoffs/NBA_{year}.html#all_all_playoffs"
    html = urlopen(url)
    soup = BeautifulSoup(html, "lxml")
    playoffs_div = soup.find("div", id="div_all_playoffs")

    def match_series(series_name: str, filters: list[str]) -> bool:
        """Determina si una serie coincide con los filtros (case-insensitive)."""
        for f in filters:
            f_lower = f.lower()
            if f_lower == "finals":  # excepción: Finals debe ser exacto
                if series_name.lower() == "finals":
                    return True
            else:
                if f_lower in series_name.lower():
                    return True
        return False

    if not games:
        # --- Series DataFrame ---
        series_rows = [row for row in playoffs_div.find_all("tr") if row.find("span", class_="tooltip opener")]

        all_series = []
        for row in series_rows:
            series_name = row.find("strong").text
            if series and not match_series(series_name, series):
                continue

            teams_text = row.find_all("td")[1].get_text(separator="|").split("|")
            team_winner = teams_text[0].strip()
            team_loser = teams_text[2].strip()
            series_result = teams_text[3].strip()

            all_series.append({
                "series": series_name,
                "winner": team_winner,
                "loser": team_loser,
                "result": series_result
            })

        df_series = pd.DataFrame(all_series)
        return df_series, df_series.to_dict(orient="records")

    else:
        # --- Games DataFrame ---
        toggle_rows = [row for row in playoffs_div.find_all("tr", class_="toggleable")]
        all_games = []

        for row in toggle_rows:
            series_name = row.find_previous_sibling("tr").find("strong").text
            if series and not match_series(series_name, series):
                continue

            table = row.find("table")
            if table:
                for g in table.find_all("tr"):
                    cells = g.find_all("td")
                    if not cells:
                        continue
                    game_number = cells[0].get_text(strip=True)
                    date = cells[1].get_text(strip=True)
                    visitor = cells[2].get_text(strip=True)
                    visitor_score = cells[3].get_text(strip=True)
                    local = cells[4].get_text(strip=True).replace("@", "").strip()
                    local_score = cells[5].get_text(strip=True)

                    all_games.append({
                        "SERIES": series_name,
                        "GAME": game_number,
                        "GAME_DATE": pd.to_datetime(date + f", {year}").strftime("%Y-%m-%d"),
                        "SCORE": f"{visitor_score}-{local_score}",
                        "AWAY": visitor,
                        "HOME": local,
                        "SCORE_AWAY": visitor_score,
                        "SCORE_HOME": local_score
                    })

        df_games = pd.DataFrame(all_games)
        return df_games, df_games.to_dict(orient="records")