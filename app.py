import streamlit as st
import requests
from openai import OpenAI
import time
from rag import search_wikivoyage
import os
from agent import run_travel_agent, refine_itinerary
import json
import pydeck as pdk
import math
from feedback import save_feedback, feedback_stats
from poi_tools import search_pois

DAY_COLORS = [
    [231, 76, 60],      # Day 1 - red
    [52, 152, 219],     # Day 2 - blue
    [46, 204, 113],     # Day 3 - green
    [243, 156, 18],     # Day 4 - orange
    [155, 89, 182],     # Day 5 - purple
    [26, 188, 156],     # Day 6 - teal
    [232, 67, 147],     # Day 7 - pink
    [149, 165, 166],    # Day 8 - gray
    [52, 73, 94],       # Day 9 - dark blue
    [211, 84, 0],       # Day 10
    [41, 128, 185],     # Day 11
    [39, 174, 96],      # Day 12
    [142, 68, 173],     # Day 13
    [241, 196, 15],     # Day 14
]



st.set_page_config(
    page_title="TripAgent",
    page_icon="✈️",
    layout="wide"
)

DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR,"app_state.json")

os.makedirs(DATA_DIR,exist_ok=True)

def load_app_state():
    if not os.path.exists(STATE_FILE):
        return None

    try:
        with open(STATE_FILE,"r",encoding="utf-8") as file:
            return json.load(file)

    except(json.JSONDecodeError, OSError):
        return None

def save_app_state(state):
    try:
        with open(STATE_FILE,"w",encoding="utf-8") as file:
            json.dump(state,file,indent=2,ensure_ascii=False)

    except OSError as error:
        st.warning(f"could not save itinerary {error}")


def build_map_points(
    itinerary,
    pois,
    selected_day="Entire Trip"
):
    points = []

    for day in itinerary["days"]:
        day_number = day["day"]

        if (
            selected_day != "Entire Trip"
            and selected_day != f"Day {day_number}"
        ):
            continue

        color = DAY_COLORS[
            (day_number - 1) % len(DAY_COLORS)
        ]

        stop_number = 1

        for block_name in [
            "morning",
            "afternoon",
            "evening"
        ]:
            for item in day[block_name]:

                poi = pois.get(item["poi_id"])

                if not poi:
                    continue

                points.append({
                    "poi_id": item["poi_id"],
                    "name": item["name"],
                    "notes": item.get(
                        "notes",
                        ""
                    ),
                    "category": poi.get(
                        "category",
                        "Unknown"
                    ),
                    "lat": poi["lat"],
                    "lon": poi["lon"],
                    "url": poi.get(
                        "url",
                        ""
                    ),
                    "day": day_number,
                    "block": block_name.title(),
                    "stop_number": stop_number,
                    "color": color
                })

                stop_number += 1

    return points


def build_route_paths(points):
    grouped_days = {}

    for point in points:
        day_number = point["day"]

        if day_number not in grouped_days:
            grouped_days[day_number] = []

        grouped_days[day_number].append(
            point
        )

    paths = []

    for day_number, day_points in grouped_days.items():

        if len(day_points) < 2:
            continue

        paths.append({
            "day": day_number,

            "path": [
                [
                    point["lon"],
                    point["lat"]
                ]
                for point in day_points
            ],

            "color": DAY_COLORS[
                (day_number - 1)
                % len(DAY_COLORS)
            ]
        })

    return paths


def calculate_map_view(points):
    if not points:
        return {
            "latitude": 0,
            "longitude": 0,
            "zoom": 2
        }

    latitudes = [
        point["lat"]
        for point in points
    ]

    longitudes = [
        point["lon"]
        for point in points
    ]

    center_lat = sum(latitudes) / len(latitudes)
    center_lon = sum(longitudes) / len(longitudes)

    if len(points) == 1:
        zoom = 14

    else:
        lat_spread = max(latitudes) - min(latitudes)
        lon_spread = max(longitudes) - min(longitudes)

        spread = max(
            lat_spread,
            lon_spread,
            0.001
        )

        zoom = math.log2(360 / spread) - 1

        zoom = max(
            3,
            min(14, zoom)
        )

    return {
        "latitude": center_lat,
        "longitude": center_lon,
        "zoom": zoom
    }

#start

if "openai_api_key" not in st.session_state:
    st.session_state.open_api_key = None

if "trip_result" not in st.session_state:
    #saved_state = load_app_state()
    #if saved_state:
        #st.session_state.trip_result = saved_state
    #else:
        #st.session_state.trip_result = None

    st.session_state.trip_result = None

if "refinement_historyt" not in st.session_state:
    st.session_state.refinement_history = []


st.title("✈️ TripAgent")

st.caption("An agentic AI travel planner using real-world POIs and travel-guide context.")


with st.sidebar:
    st.header("Settings")

    api_key = st.text_input("OpenAI API Key",type="password",placeholder="sk-...")

    if st.button("Save API Key",use_container_width=True):
        if api_key:
            st.session_state.openai_api_key = api_key

            st.success("API key saved for this session.")

        else:
            st.warning("Enter an API key first.")


    if st.button("Clear API Key",use_container_width=True):
        st.session_state.openai_api_key = None
        st.success("API key cleared.")

    st.divider()

    st.subheader("Generation Settings")

    fast_mode = st.toggle(
        "⚡ Fast Mode",
        value=False,
        help=(
            "Uses fewer AI tool calls for faster itinerary generation. "
            "May do less destination research."
        )
    )

    model_choice = st.selectbox(
        "Model",
        [
            "gpt-5.6",
            "gpt-5.6-mini"
        ],
        index=0,
        help="Choose the AI model used to generate the itinerary."
    )

    if fast_mode:
        max_steps = 5

        st.caption(
            "Fast Mode: up to 5 agent steps."
        )

    else:
        max_steps = st.slider(
            "Maximum agent steps",
            min_value=5,
            max_value=12,
            value=8,
            help=(
                "Higher values allow more research/tool calls "
                "but take longer."
            )
        )


# Trip inputs

st.header("Plan your trip")


col1, col2 = st.columns(2)


with col1:

    destination = st.text_input("Destination",placeholder="Tokyo",help=(        "Enter a city or destination such as Tokyo, Paris, or Santa Fe."))

    trip_length = st.number_input("Trip length",min_value=1,max_value=14,value=3,step=1,help="Number of days to include in the itinerary. Relaxed means fewer stops; fast-paced means more activities each day.")

with col2:

    pace = st.selectbox("Travel pace",["Relaxed","Balanced","Fast-paced"],index=1)

    interests = st.multiselect("Interests",["Food","Museums","History","Outdoors","Shopping","Nightlife","Architecture","Religious sites","Scenic Views","Local Culture"],default=["Food","History"],help = ("Choose the types of places the agent should prioritize."))


constraints = st.text_area("Constraints or preferences",
    placeholder=(
        "Examples: vegetarian food, avoid long walks, "
        "traveling with children, budget-conscious, "
        "no early mornings..."
    ),
    help =("Add accessibility needs, dietary preferences, budget considerations, or scheduling preferences.")
)

generate = st.button("✨ Generate Itinerary",type="primary",use_container_width=True)

if generate:

    destination = destination.strip()

    if not st.session_state.openai_api_key:

        st.error(
            "Enter and save your OpenAI API key first."
        )

    elif not destination:

        st.warning(
            "Enter a destination."
        )

    elif len(destination) < 2:

        st.warning(
            "Destination name is too short."
        )

    elif trip_length < 1 or trip_length > 14:

        st.warning(
            "Trip length must be between 1 and 14 days."
        )

    elif not interests:

        st.warning(
            "Choose at least one interest."
        )

    elif len(constraints) > 2000:

        st.warning(
            "Please keep constraints under 2000 characters."
        )
    else:

        user_request = f"""
        Plan a {trip_length}-day trip to {destination}.

        Travel pace: {pace}

        Interests:
        {", ".join(interests)}

        Additional constraints:
        {constraints if constraints.strip() else "None"}

        Create a practical itinerary organized into morning,
        afternoon, and evening activities.
        """

        try:

            with st.status("Planning your trip...",expanded=True) as status:
                def update_progress(message):
                    status.write(message)
                generation_start = time.perf_counter()
                result = run_travel_agent(
                    api_key=(
                        st.session_state.openai_api_key
                    ),
                    user_request=user_request,
                    progress_callback=update_progress,
                    max_steps=max_steps,
                    model=model_choice,
                    fast_mode=fast_mode
                )
                generation_seconds = (time.perf_counter() - generation_start)
                result['generation_seconds'] = round(generation_seconds, 2)

                status.update(label="Itinerary ready!",state="complete",expanded=False)

            st.session_state.trip_result = result
            #save_app_state(result)

        except ValueError as error:

            st.error(
                f"Could not create a valid itinerary: {error}"
            )

            st.info(
                "Try a different destination, broader interests, "
                "or fewer constraints."
            )


        except RuntimeError as error:

            st.error(
                str(error)
            )


        except Exception as error:

            st.error(
                "Something unexpected went wrong."
            )

            with st.expander(
                "Technical details"
            ):

                st.code(
                    str(error)
                )


#Display Itin


result = st.session_state.trip_result


if result:

    itinerary = result["itinerary"]
    tool_state = result["tool_state"]

    pois = tool_state.get(
        "pois",
        {}
    )

    st.divider()

    st.header(
        f"🗺️ {itinerary['destination']}"
    )


    if result.get("generation_seconds"):
        st.caption(
            f"Generated in "
            f"{result['generation_seconds']:.1f} seconds"
        )

        
    total_stops = sum(
        len(day["morning"])
        + len(day["afternoon"])
        + len(day["evening"])
        for day in itinerary["days"]
    )

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Days",
        len(itinerary["days"])
    )

    metric2.metric(
        "Stops",
        total_stops
    )

    metric3.metric(
        "POIs researched",
        len(pois)
    )



    for day in itinerary["days"]:

        st.subheader(
            f"Day {day['day']}"
        )

        morning_col, afternoon_col, evening_col = (
            st.columns(3)
        )


        blocks = [
            (
                morning_col,
                "🌅 Morning",
                day["morning"]
            ),
            (
                afternoon_col,
                "☀️ Afternoon",
                day["afternoon"]
            ),
            (
                evening_col,
                "🌙 Evening",
                day["evening"]
            )
        ]


        for column, heading, activities in blocks:

            with column:

                st.markdown(
                    f"### {heading}"
                )

                if not activities:

                    st.caption(
                        "No activity planned."
                    )

                for item in activities:

                    poi_id = item["poi_id"]

                    poi = pois.get(
                        poi_id,
                        {}
                    )

                    st.markdown(
                        f"**{item['name']}**"
                    )


                    if poi.get("category"):

                        st.caption(
                            "Category: "
                            f"{poi['category'].title()}"
                        )


                    st.write(
                        item["notes"]
                    )


                    if poi.get("url"):

                        st.markdown(
                            f"[View on OpenStreetMap]"
                            f"({poi['url']})"
                        )


                    # -------------------------
                    # FEEDBACK
                    # -------------------------

                    feedback_col1, feedback_col2 = (
                        st.columns(2)
                    )

                    with feedback_col1:

                        if st.button(
                            "👍",
                            key=(
                                f"up_{day['day']}_"
                                f"{heading}_{poi_id}"
                            )
                        ):

                            save_feedback(
                                itinerary["destination"],
                                poi_id,
                                "up"
                            )

                            search_pois.clear()

                            st.toast(
                                "Feedback saved 👍"
                            )


                    with feedback_col2:

                        if st.button(
                            "👎",
                            key=(
                                f"down_{day['day']}_"
                                f"{heading}_{poi_id}"
                            )
                        ):

                            save_feedback(
                                itinerary["destination"],
                                poi_id,
                                "down"
                            )

                            search_pois.clear()

                            st.toast(
                                "Feedback saved 👎"
                            )


                    st.caption(
                        f"POI ID: {poi_id}"
                    )

                    st.divider()
    # =========================================================
    # INTERACTIVE MAP
    # =========================================================

    st.subheader("🗺️ Itinerary Map")

    st.caption(
        "Hover over a stop for details. "
        "Select a specific day to see its route "
        "and numbered stop order."
    )


    # =====================================================
    # MAP CONTROLS
    # =====================================================

    day_options = [
        "Entire Trip"
    ] + [
        f"Day {day['day']}"
        for day in itinerary["days"]
    ]

    map_col1, map_col2 = st.columns(
        [2, 1]
    )

    with map_col1:

        selected_day = st.selectbox(
            "Show on map",
            day_options,
            key="map_day_filter",
            help=(
                "View the entire trip or focus on "
                "one day's route."
            )
        )


    with map_col2:

        map_theme = st.selectbox(
            "Map style",
            [
                "Light",
                "Dark"
            ],
            key="map_theme",
            help="Choose the map appearance."
        )


    # =====================================================
    # BUILD MAP DATA
    # =====================================================

    map_points = build_map_points(
        itinerary,
        pois,
        selected_day
    )

    route_paths = build_route_paths(
        map_points
    )


    if not map_points:

        st.info(
            "No mappable locations were found "
            "for this selection."
        )

    else:

        # =================================================
        # LEGEND
        # =================================================
        
        visible_days = sorted(
            set(
                point["day"]
                for point in map_points
            )
        )

        legend_items = []

        for day_number in visible_days:

            color = DAY_COLORS[
                (day_number - 1) % len(DAY_COLORS)
            ]

            hex_color = (
                f"#{color[0]:02x}"
                f"{color[1]:02x}"
                f"{color[2]:02x}"
            )

            legend_items.append(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    gap:6px;
                    margin-right:18px;
                ">
                    <div style="
                        width:12px;
                        height:12px;
                        border-radius:50%;
                        background:{hex_color};
                        flex-shrink:0;
                    "></div>

                    <span style="
                        font-size:14px;
                    ">
                        Day {day_number}
                    </span>
                </div>
                """
            )

        legend_html = f"""
        <div style="
            display:flex;
            flex-wrap:wrap;
            align-items:center;
            gap:8px;
            margin-top:4px;
            margin-bottom:12px;
        ">
            {''.join(legend_items)}
        </div>
        """

        st.html(legend_html)



        # =================================================
        # POINT LAYER
        # =================================================

        poi_layer = pdk.Layer(
            "ScatterplotLayer",

            data=map_points,

            get_position="[lon, lat]",

            get_fill_color="color",

            get_line_color=[
                255,
                255,
                255,
                230
            ],

            get_radius=50,

            radius_min_pixels=7,

            radius_max_pixels=12,

            line_width_min_pixels=2,

            stroked=True,

            filled=True,

            pickable=True,

            auto_highlight=True
        )


        # =================================================
        # NUMBER LABELS
        # =================================================

        label_layer = pdk.Layer(
            "TextLayer",

            data=map_points,

            get_position="[lon, lat]",

            get_text="stop_number",

            get_size=12,

            get_color=[
                255,
                255,
                255
            ],

            get_text_anchor='"middle"',

            get_alignment_baseline='"center"',

            pickable=False
        )


        # =================================================
        # ROUTE LAYER
        # =================================================

        route_layer = pdk.Layer(
            "PathLayer",

            data=route_paths,

            get_path="path",

            get_color="color",

            get_width=4,

            width_min_pixels=2,

            width_max_pixels=4,

            pickable=False
        )


        # =================================================
        # MAP STYLE
        # =================================================

        if map_theme == "Dark":

            map_style = (
                "https://basemaps.cartocdn.com/"
                "gl/dark-matter-gl-style/style.json"
            )

        else:

            map_style = (
                "https://basemaps.cartocdn.com/"
                "gl/positron-gl-style/style.json"
            )


        # =================================================
        # VIEW
        # =================================================

        view_state = calculate_map_view(
            map_points
        )


        # =================================================
        # TOOLTIP
        # =================================================

        tooltip = {
            "html": """
            <div style="
                min-width:220px;
                max-width:300px;
            ">

                <div style="
                    font-size:16px;
                    font-weight:600;
                    margin-bottom:5px;
                ">
                    {name}
                </div>

                <div>
                    Day {day} · Stop {stop_number}
                </div>

                <div>
                    {block} · {category}
                </div>

                <div style="
                    margin-top:8px;
                    font-size:12px;
                    opacity:0.9;
                ">
                    {notes}
                </div>

            </div>
            """,

            "style": {
                "backgroundColor": "#1f2937",
                "color": "white"
            }
        }


        # =================================================
        # IMPORTANT:
        # ENTIRE TRIP = NO ROUTE SPAGHETTI
        # SINGLE DAY = SHOW ROUTE
        # =================================================

        if selected_day == "Entire Trip":

            map_layers = [
                poi_layer,
                label_layer
            ]

        else:

            map_layers = [
                route_layer,
                poi_layer,
                label_layer
            ]


        # =================================================
        # DRAW MAP
        # =================================================

        deck = pdk.Deck(
            layers=map_layers,
            initial_view_state=view_state,
            map_style=map_style,
            tooltip=tooltip
        )

        st.pydeck_chart(
            deck,
            use_container_width=True
        )

    st.divider()
    st.header("✏️ Refine Itinerary")

    refinement_request = st.text_input(
        "What would you like to change?",
        placeholder=(
            "Make it more outdoorsy, add more food stops, "
            "make the mornings more relaxed..."
        ),
        key="refinement_request"
    )

    refinement_mode = st.radio(
        "What should be changed?",
        [
            "Entire itinerary",
            "Single day"
        ],
        horizontal=True
    )

    target_day = None

    if refinement_mode == "Single day":
        target_day = st.selectbox(
            "Day to regenerate",
            [
                day["day"]
                for day in itinerary["days"]
            ],
            format_func=lambda day: f"Day {day}",
            key="refinement_target_day"
        )

    if st.button(
        "Refine Itinerary",
        type="primary",
        use_container_width=True
    ):

        if not refinement_request.strip():
            st.warning(
                "Enter a refinement request first."
            )

        elif not st.session_state.openai_api_key:
            st.error(
                "Enter and save your OpenAI API key first."
            )

        else:
            old_itinerary = itinerary

            try:
                with st.status(
                    "Refining itinerary...",
                    expanded=True
                ) as status:

                    if target_day is None:
                        status.write(
                            "Refining the full itinerary..."
                        )
                    else:
                        status.write(
                            f"Regenerating only Day {target_day}..."
                        )

                    refined_itinerary = refine_itinerary(
                        api_key=st.session_state.openai_api_key,
                        itinerary=old_itinerary,
                        tool_state=tool_state,
                        user_request=refinement_request,
                        target_day=target_day
                    )

                    status.update(
                        label="Refinement complete!",
                        state="complete",
                        expanded=False
                    )

                st.session_state.refinement_history.append({
                    "request": refinement_request,
                    "target_day": target_day,
                    "before": old_itinerary,
                    "after": refined_itinerary
                })

                st.session_state.trip_result["itinerary"] = (
                    refined_itinerary
                )

                save_app_state(
                    st.session_state.trip_result
                )

                st.rerun()

            except Exception as error:
                st.error(
                    f"Refinement failed: {error}"
                )


            
    guide_chunks = tool_state.get(
        "guide_chunks",
        []
    )

    if guide_chunks:

        st.subheader(
            "📚 Sources"
        )

        unique_sources = []

        for chunk in guide_chunks:

            source = chunk.get(
                "source"
            )

            if (
                source
                and source not in unique_sources
            ):
                unique_sources.append(
                    source
                )


        for source in unique_sources:

            st.markdown(
                f"- [Wikivoyage travel guide]"
                f"({source})"
            )


        with st.expander(
            "View retrieved travel-guide context"
        ):

            for chunk in guide_chunks:

                score = chunk.get(
                    "score",
                    0
                )

                st.caption(
                    f"Relevance score: {score:.3f}"
                )

                st.write(
                    chunk.get(
                        "text",
                        ""
                    )
                )

                st.divider()

    if st.session_state.refinement_history:

        with st.expander(
            "🕘 Refinement History"
        ):

            for index, change in enumerate(
                reversed(
                    st.session_state.refinement_history
                ),
                start=1
            ):

                st.markdown(
                    f"### Change {index}"
                )

                st.write(
                    f"**Request:** {change['request']}"
                )

                if change["target_day"]:
                    st.caption(
                        f"Only Day {change['target_day']} regenerated"
                    )
                else:
                    st.caption(
                        "Full itinerary refinement"
                    )

                before_col, after_col = st.columns(2)

                with before_col:
                    st.markdown("**Before**")
                    st.json(
                        change["before"],
                        expanded=False
                    )

                with after_col:
                    st.markdown("**After**")
                    st.json(
                        change["after"],
                        expanded=False
                    )

                st.divider()

# =========================================================
# AGENT TRACE
# =========================================================

    with st.expander("🤖 View agent activity"):

        for event in tool_state.get(
            "trace",
            []
        ):

            step = event.get(
                "step",
                "?"
            )

            action = event.get(
                "action",
                ""
            )

            tool = event.get(
                "tool"
            )
            duration = event.get(
                "duration_seconds"
            )

            if tool:

                st.write(
                    f"**Step {step}:** "
                    f"{action} — `{tool}`"
                )

            else:

                st.write(
                    f"**Step {step}:** "
                    f"{action}"
                )

            if event.get(
                "arguments"
            ):

                st.json(
                    event["arguments"]
                )
            if duration is not None:
                st.caption(
                    f"⏱️ {duration:.2f} seconds"
                )


# =========================================================
# DOWNLOAD
# =========================================================

    st.subheader(
        "Export"
    )

    itinerary_json = json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    )

    st.download_button(
        label="⬇️ Download Itinerary JSON",
        data=itinerary_json,
        file_name="trip_itinerary.json",
        mime="application/json",
        use_container_width=True
    )


# =========================================================
# CLEAR SAVED TRIP
# =========================================================

    if st.button(
        "Clear Saved Itinerary"
    ):

        st.session_state.trip_result = None

        if os.path.exists(
            STATE_FILE
        ):
            os.remove(
                STATE_FILE
            )

        st.rerun()

