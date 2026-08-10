import json

from openai import (OpenAI, APITimeoutError, RateLimitError, APIConnectionError, APIError)

import time
from poi_tools import search_pois, geocode_city
from rag import retrieve_guides


TOOLS = [
    {
        "type": "function",
        "name": "search_pois",
        "description": (
            "Search OpenStreetMap for real points of interest in a city "
            "based on a traveler's interest category."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string"
                },
                "interest": {
                    "type": "string",
                    "enum": [
                        "museums",
                        "food",
                        "outdoors",
                        "history",
                        "shopping"
                    ]
                },
                "radius": {
                    "type": "integer"
                }
            },
            "required": [
                "city",
                "interest",
                "radius"
            ],
            "additionalProperties": False
        },
        "strict": True
    },

    {
        "type": "function",
        "name": "retrieve_guides",
        "description": (
            "Retrieve relevant Wikivoyage guide passages "
            "for a destination."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {
                    "type": "string"
                },
                "query": {
                    "type": "string"
                },
                "top_k": {
                    "type": "integer"
                }
            },
            "required": [
                "destination",
                "query",
                "top_k"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
]

ITINERARY_SCHEMA = {
    "type": "object",
    "properties": {
        "destination": {
            "type": "string"
        },
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "integer"
                    },
                    "morning": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "poi_id": {"type": "string"},
                                "name": {"type": "string"},
                                "notes": {"type": "string"}
                            },
                            "required": [
                                "poi_id",
                                "name",
                                "notes"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "afternoon": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "poi_id": {"type": "string"},
                                "name": {"type": "string"},
                                "notes": {"type": "string"}
                            },
                            "required": [
                                "poi_id",
                                "name",
                                "notes"
                            ],
                            "additionalProperties": False
                        }
                    },
                    "evening": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "poi_id": {"type": "string"},
                                "name": {"type": "string"},
                                "notes": {"type": "string"}
                            },
                            "required": [
                                "poi_id",
                                "name",
                                "notes"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": [
                    "day",
                    "morning",
                    "afternoon",
                    "evening"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "destination",
        "days"
    ],
    "additionalProperties": False
}




AGENT_INSTRUCTIONS = """
You are an AI travel-planning agent.

Use search_pois to discover real places.
Use retrieve_guides for destination context.

Never invent POIs.

Every specific attraction, restaurant, museum, park,
or other itinerary stop must come from search_pois.

When mentioning a POI in the itinerary, include its exact poi_id.

Use tools until you have enough information, then produce
a concise day-by-day itinerary.

- Write all itinerary notes and descriptions in English.
- Prefer English POI names when provided.
- If a POI only has a local-language name, preserve that exact name rather than inventing a translation.
"""


def create_tool_state():
    return {
        "pois": {},
        "guide_chunks": [],
        "city_metadata": {},
        "trace": []
    }


def execute_tool(
    function_name,
    arguments,
    tool_state
):
    if function_name == "search_pois":
        
        city = arguments["city"]

        pois = search_pois(
            city=city,
            interest=arguments["interest"],
            radius=arguments["radius"]
        )

        for poi in pois:
            tool_state["pois"][poi["poi_id"]] = poi

        city_data = geocode_city(city)

        if city_data:
            tool_state["city_metadata"][city] = city_data

        return pois

    elif function_name == "retrieve_guides":

        try:

            chunks = retrieve_guides(
                destination=arguments["destination"],
                query=arguments["query"],
                top_k=arguments["top_k"]
            )

        except Exception as error:

            tool_state["trace"].append({
                "action": "Guide retrieval failed",
                "error": str(error)
            })

            return []

        tool_state["guide_chunks"].extend(
            chunks
        )

        return chunks

    raise ValueError(
        f"Unknown tool: {function_name}"
    )

def run_travel_agent(
    api_key,
    user_request,
    max_steps=8,
    progress_callback=None,
    model="gpt-5.6",
    fast_mode=False
):
    client = OpenAI(
        api_key=api_key,
        timeout=60.0,
        max_retries=2
    )

    if fast_mode:
        mode_instructions = """
FAST MODE IS ENABLED.

- Minimize tool calls.
- Prefer one broad search_pois call when possible.
- Retrieve only essential guide context.
- Stop researching once enough POIs exist to build the itinerary.
- Prioritize speed over exhaustive research.
"""
    else:
        mode_instructions = """
NORMAL MODE IS ENABLED.

- Research the destination thoroughly.
- You may call search_pois multiple times for different interests.
- Retrieve relevant Wikivoyage context when useful.
- Stop once you have enough information for a strong itinerary.
"""

    tool_state = create_tool_state()

    input_items = [
        {
            "role": "user",
            "content": user_request
        }
    ]

    for step in range(max_steps):

        step_start = time.perf_counter()

        tool_state["trace"].append({
            "step": step + 1,
            "action": "Calling model"
        })

        if progress_callback:
            progress_callback(
                f"🧠 Step {step + 1}/{max_steps}: "
                "Asking the AI what to do next..."
            )

        # =====================================================
        # CALL MODEL
        # =====================================================

        try:
            response = client.responses.create(
                model=model,
                instructions=(
                    AGENT_INSTRUCTIONS
                    + "\n"
                    + mode_instructions
                ),
                tools=TOOLS,
                input=input_items
            )

        except RateLimitError:
            raise RuntimeError(
                "OpenAI is rate limiting requests. "
                "Wait a moment and try again."
            )

        except APITimeoutError:
            raise RuntimeError(
                "The AI request took too long. "
                "Please try again."
            )

        except APIConnectionError:
            raise RuntimeError(
                "Could not connect to OpenAI. "
                "Check your internet connection."
            )

        except APIError as error:
            raise RuntimeError(
                f"OpenAI API error: {error}"
            )


        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]


        # =====================================================
        # NO MORE TOOLS -> GENERATE FINAL ITINERARY
        # =====================================================

        if not function_calls:

            tool_state["trace"].append({
                "step": step + 1,
                "action": "Generating validated itinerary"
            })

            if not tool_state["pois"]:
                raise ValueError(
                    "No suitable places were found for this trip. "
                    "Try a larger nearby city or broader interests."
                )

            allowed_pois = list(
                tool_state["pois"].values()
            )

            guide_chunks = tool_state["guide_chunks"]

            final_prompt = f"""
Create the final itinerary now.

You may ONLY use POIs from the following list:

{json.dumps(allowed_pois)}

Here is relevant travel-guide context retrieved from Wikivoyage:

{json.dumps(guide_chunks)}

RULES:

- Every itinerary item MUST use an exact poi_id from the allowed POI list.
- Do not invent POIs.
- Do not use a place mentioned only in the Wikivoyage context unless it also exists in the allowed POI list.
- Use the Wikivoyage context for useful background, planning advice, and destination context.
- Make the itinerary practical and organized by morning, afternoon, and evening.
- Write all itinerary notes and descriptions in English.
- Prefer English POI names when provided.
- If a POI only has a local-language name, preserve that exact name rather than inventing a translation.
"""

            if progress_callback:
                progress_callback(
                    "✨ Research complete. Building your itinerary..."
                )

            try:
                final_response = client.responses.create(
                    model=model,
                    instructions=(
                        AGENT_INSTRUCTIONS
                        + "\n"
                        + mode_instructions
                    ),
                    input=[
                        {
                            "role": "user",
                            "content": user_request
                        },
                        {
                            "role": "user",
                            "content": final_prompt
                        }
                    ],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "travel_itinerary",
                            "schema": ITINERARY_SCHEMA,
                            "strict": True
                        }
                    }
                )

            except RateLimitError:
                raise RuntimeError(
                    "OpenAI is rate limiting requests. "
                    "Wait a moment and try again."
                )

            except APITimeoutError:
                raise RuntimeError(
                    "The final itinerary request took too long. "
                    "Please try again."
                )

            except APIConnectionError:
                raise RuntimeError(
                    "Could not connect to OpenAI."
                )

            except APIError as error:
                raise RuntimeError(
                    f"OpenAI API error: {error}"
                )


            raw_output = final_response.output_text

            try:
                itinerary = json.loads(
                    raw_output
                )

            except json.JSONDecodeError as error:
                raise ValueError(
                    "The model returned an itinerary that "
                    f"could not be parsed as JSON: {error}"
                )


            validate_itinerary_poi_ids(
                itinerary,
                tool_state["pois"]
            )


            step_duration = (
                time.perf_counter()
                - step_start
            )

            tool_state["trace"].append({
                "step": step + 1,
                "action": "Itinerary validated",
                "poi_count": len(
                    tool_state["pois"]
                ),
                "guide_chunk_count": len(
                    tool_state["guide_chunks"]
                ),
                "duration_seconds": round(
                    step_duration,
                    2
                )
            })


            if progress_callback:
                progress_callback(
                    "✅ Itinerary complete and validated."
                )


            return {
                "itinerary": itinerary,
                "tool_state": tool_state
            }


        # =====================================================
        # TOOL CALLS
        # =====================================================

        input_items += response.output


        for call in function_calls:

            tool_start = time.perf_counter()


            # -------------------------------------------------
            # PARSE TOOL ARGUMENTS
            # -------------------------------------------------

            try:
                arguments = json.loads(
                    call.arguments
                )

            except json.JSONDecodeError:

                tool_state["trace"].append({
                    "step": step + 1,
                    "action": "Invalid tool arguments",
                    "tool": call.name,
                    "raw_arguments": call.arguments
                })

                result = {
                    "error": (
                        "The model returned invalid tool arguments."
                    )
                }

                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result)
                })

                continue


            tool_state["trace"].append({
                "step": step + 1,
                "action": "Tool call",
                "tool": call.name,
                "arguments": arguments
            })


            # -------------------------------------------------
            # LIVE STATUS MESSAGE
            # -------------------------------------------------

            if progress_callback:

                if call.name == "search_pois":
                    progress_callback(
                        "📍 Searching OpenStreetMap for places..."
                    )

                elif call.name == "retrieve_guides":
                    progress_callback(
                        "📚 Retrieving Wikivoyage travel context..."
                    )

                else:
                    progress_callback(
                        f"Using {call.name}..."
                    )


            # -------------------------------------------------
            # EXECUTE TOOL
            # -------------------------------------------------

            try:

                result = execute_tool(
                    call.name,
                    arguments,
                    tool_state
                )

                tool_duration = (
                    time.perf_counter()
                    - tool_start
                )

                result_count = (
                    len(result)
                    if hasattr(result, "__len__")
                    else 1
                )

                tool_state["trace"].append({
                    "step": step + 1,
                    "action": "Tool completed",
                    "tool": call.name,
                    "result_count": result_count,
                    "duration_seconds": round(
                        tool_duration,
                        2
                    )
                })


                if progress_callback:
                    progress_callback(
                        f"✅ {call.name} completed "
                        f"in {tool_duration:.1f}s."
                    )


            except Exception as error:

                tool_duration = (
                    time.perf_counter()
                    - tool_start
                )

                result = {
                    "error": str(error)
                }

                tool_state["trace"].append({
                    "step": step + 1,
                    "action": "Tool error",
                    "tool": call.name,
                    "error": str(error),
                    "duration_seconds": round(
                        tool_duration,
                        2
                    )
                })


                if progress_callback:
                    progress_callback(
                        f"⚠️ {call.name} failed: {error}"
                    )


            # -------------------------------------------------
            # RETURN TOOL OUTPUT TO MODEL
            # -------------------------------------------------

            input_items.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(result)
            })


    raise RuntimeError(
        f"Agent exceeded maximum step limit of {max_steps}."
    )


def verify_other_days_unchanged(
    original,
    refined,
    target_day
):
    original_days = {
        day["day"]: day
        for day in original["days"]
    }

    refined_days = {
        day["day"]: day
        for day in refined["days"]
    }

    if set(original_days.keys()) != set(refined_days.keys()):
        raise ValueError(
            "Refinement changed the number or identity of trip days."
        )

    for day_number in original_days:

        if day_number == target_day:
            continue

        if original_days[day_number] != refined_days[day_number]:
            raise ValueError(
                f"Day {day_number} changed even though only "
                f"Day {target_day} was supposed to change."
            )

    return True







def refine_itinerary(
    api_key,
    itinerary,
    tool_state,
    user_request,
    target_day=None
):
    client = OpenAI(api_key=api_key,timeout=60.0, max_retries=2)

    allowed_pois = list(
        tool_state["pois"].values()
    )

    if target_day is None:
        prompt = f"""
Refine the existing itinerary based on the user's request.

Existing itinerary:
{json.dumps(itinerary)}

User request:
{user_request}

You may ONLY use POIs from this allowed list:
{json.dumps(allowed_pois)}

Rules:
- Preserve the overall trip structure unless the request requires a change.
- Every itinerary item must use an exact poi_id from the allowed POI list.
- Do not invent POIs.
- Return the full itinerary.
"""
    else:
        prompt = f"""
Goal: ONLY modify day {target_day}.
All other days must remain EXACTLY unchanged.

Existing itinerary:
{json.dumps(itinerary)}

User request:
{user_request}

You may ONLY use POIs from this allowed list:
{json.dumps(allowed_pois)}

Rules:
- Return the FULL itinerary.
- Only day {target_day} may change.
- Every other day must be byte-for-byte equivalent in structure and values.
- Every itinerary item must use an exact poi_id from the allowed POI list.
- Do not invent POIs.
"""

    response = client.responses.create(
        model="gpt-5.6",
        input=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "travel_itinerary",
                "schema": ITINERARY_SCHEMA,
                "strict": True
            }
        }
    )

    refined = json.loads(
        response.output_text
    )

    validate_itinerary_poi_ids(
        refined,
        tool_state["pois"]
    )

    if target_day is not None:
        verify_other_days_unchanged(
            itinerary,
            refined,
            target_day
        )

    return refined




def validate_itinerary_poi_ids(itinerary, allowed_pois):
    valid_ids = set(allowed_pois.keys())

    for day in itinerary["days"]:
        for block in [
            "morning",
            "afternoon",
            "evening"
        ]:
            for item in day[block]:
                poi_id = item["poi_id"]

                if poi_id not in valid_ids:
                    raise ValueError(
                        f"Invalid poi_id: {poi_id}"
                    )

    return True