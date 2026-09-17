"""
Docstring for data.osm_to_pp

TODO:
 [ ] Sphinx-ify all docstrings
 [ ] Graph breadth-first search to expand OSM pp data past power station, see transnet (it does this)
 [ ] End goal: be able to do OSM --> Network() where we can traverse OSM data dynamically (ofc store in cache for later easy access if computationally diffucult, w/ graph BFS hyperparameters)



"""

import requests
import xml.etree.ElementTree as ET
import queue
import math
import re
import json
from collections import defaultdict
import pandapower as pp
import pandapower.auxiliary as aux
import sys,os
# import module from enclosing directory
cdir = sys.path[0] + '/..'
if cdir not in sys.path:
    sys.path.append(cdir)
from util import timeIt

@timeIt
def get_OSM_data(bbox, KEYTAG='power', type_info=True, round_to=7, print_info=True)\
    -> tuple[dict[int,dict], dict[str,list], dict[str,dict]] | tuple[dict[int,dict], dict[str,list]]:
    """ 
    Downloads open street map data within the specified bounding box using the OSM API.
    Only filters items that have the KEYTAG tag. 
    
     INPUT: 
      bbox: tuple of 4 floats (left, bottom, right, top). needed to define rectangular bounding box
      round_to: int. specifies the number of decimal places to round the bounding box to
      print_info: bool. if True, print the number of nodes, ways, and relations found in the bounding box
      type_info: bool. if True, make and return a SUBTAGS dictionary that keeps track of the number of 
                       occurences for each tag within each power type. 
                       allows later printing with print_osm_info below.
      KEYTAG: str. the tag to filter by. default is 'power' bc we care about the electrical grid here.
    
     OUTPUT: a 2 or 3-tuple of DATA dictionary, INDICES dictionary, and maybe SUBTAGS dictionary
      DATA is a dictionary of all KEYTAG related items in the bounding box, so that 
      DATA[id] = {'osm_type': 'node', 'lat': lat, 'lon': lon, tags...}         or
                 {'osm_type': 'way', 'nodes': [n1, n2, ...],... tags}     or 
                 {'osm_type': 'relation', 'members': {'nodes': [n1, n2, ...],'ways': [w1, w2, ...]... },... }
        TODO: better way to store relations?
      INDICES is a dictionary of all KEYTAG tags with their indices in the dictionary.
        example for all tems with a tag that has 'k'=KEYTAG, 'v'='generator':
          INDICES['generator'] = [id1, id2, ...] 
      SUBTAGS is a dictionary of all KEYTAG tags with their subtags and the number of occurences of each subtag.
         this is used for later information printing with print_osm_info below.
         example:   SUBTAGS['generator'] = {'gen:type': {'wind':3,'solar':2}, 'name': {}, ...}
    """
    # store all power data in dictionary
    DATA = {}
    # keep track of all indices with each power tag 
    # to easily access all power plants, for example
    INDICES = defaultdict(list)
    # keep track of the type of each power tag. some things might have 
    # multiple types.
    # keep track of the number of occurences for each tag within 
    # each power type
    if type_info:
        SUBTAGS = defaultdict(dict)

    def update_info_dicts(id, tags):
        ''' update INDICES and SUBTAGS '''
        ptype = tags[KEYTAG]
        INDICES[ptype].append(id)
        # keep track of the number of occurences for each tag within a power type
        if type_info:
            for key, val in tags.items():
                if key not in (KEYTAG, 'lat', 'lon', 'nodes', 'members', 'nnodes'):
                    SUBTAGS[ptype][key] = SUBTAGS[ptype].get(key, defaultdict(int))
                    SUBTAGS[ptype][key][val] += 1
    
    bbox = tuple(map(lambda x: round(x, round_to), bbox))
    Q = queue.Queue()
    Q.put(bbox)
    nbboxes = 1

    nnodes = nways = nrels = 0
    while nbboxes:
        bbox_ = Q.get()
        nbboxes -= 1
        # api request for selected bounding box
        url = "https://api.openstreetmap.org/api/0.6/map?bbox="
        url += ",".join(map(str, bbox_))
        #print(url)
        request = requests.get(url)   # make the api request
        response = request.content    # get the (bytes) response
        try:   # parse the response into a dictionary
            root = ET.fromstring(response)
        except ET.ParseError:
            # we requested too many nodes, so now let's
            # split the bounding box into 4 quadrants
            left, bottom, right, top = bbox_
            mid_x = round((left + right) / 2,round_to)
            mid_y = round((bottom + top) / 2,round_to)
            # add 4 quadrants to the queue
            Q.put((left, bottom, mid_x, mid_y))  # bottom left
            Q.put((mid_x, bottom, right, mid_y)) # bottom right
            Q.put((left, mid_y, mid_x, top))     # top left
            Q.put((mid_x, mid_y, right, top))    # top right
            nbboxes += 4
            continue
        
        # add all KEYTAG related items to dictionary for later use
        for element in root:
            # filter only items with KEYTAG specified
            save = False
            for child in element:
                if child.tag == 'tag':
                    if child.get('k') == KEYTAG:
                        save = True
                        break
            if not save: continue

            # parse element and save to dictionaries
            id = int(element.get('id'))
            if element.tag == 'node':
                tags = {"osm_type": "node", "lat": element.get('lat'), "lon": element.get('lon')}
                for child in element:   # assume all tages are child.tag == 'tag'
                    if child.tag == 'tag':
                        key = child.get('k'); val = child.get('v')
                        if key and val:
                            tags[key] = val
                            if key == KEYTAG: save = True
                        else:
                            print(f"Node {id} has a tag with no key-value: ", child.attrib)
                nnodes += 1
            elif element.tag == 'way':
                tags = {'osm_type': "way", 'nodes': []}
                waylen = 0
                for child in element:
                    if child.tag == 'nd':
                        tags['nodes'].append(child.get('ref'))
                        waylen += 1
                    elif child.tag == 'tag':
                        key = child.get('k'); val = child.get('v')
                        if key and val:
                            tags[key] = val
                        else:
                            print(f"Way {id} has a tag with no key-value: ", child.attrib)
                tags['nnodes'] = waylen
                nways += 1
            elif element.tag == 'relation':
                tags = {"osm_type": "relation", 'members': {}}
                for member in element:
                    if member.tag == 'member':
                        key = member.get('type')+'s'
                        if key not in tags['members']:
                            tags['members'][key] = []
                        tags['members'][key].append(member.get('ref'))
                    elif member.tag == 'tag':
                        key = child.get('k'); val = child.get('v')
                        if key and val:
                            tags[key] = val
                        else:
                            print(f"Relation {id} has a tag with no key-value: ", child.attrib)
                
                nrels += 1
            else:
                print("Unknown element: ", element)
                continue
            DATA[id] = tags
            update_info_dicts(id, tags)
    if print_info:
        print(f"Bbox {bbox} has {nnodes} nodes, {nways} ways, and {nrels} relations.")
    if type_info:
        return DATA, INDICES, SUBTAGS
    return DATA, INDICES


def print_osm_info(INDICES, SUBTAGS, KEYTAG='power', indent=4):
    """ prettily prints all openstreetmap information. 
        ensures right-indentation of counts for prettiness via the largest 
        count of a single KEYTAG instance """
    keyinfo = [(key, len(INDICES[key])) for key in SUBTAGS.keys()]
    keyinfo.sort(key=lambda x: x[1], reverse=True)
    alen = len(str(max([count for key, count in keyinfo])))
    for x in keyinfo:
        key, nitems = x
        lev0 = indent + alen - len(str(nitems))
        print(' '*lev0+f"{nitems} {KEYTAG}={key} has {len(SUBTAGS[key])} diff tags:")
        taginfo = [(tag, tagdict, sum(c for c in tagdict.values())) for tag, tagdict in SUBTAGS[key].items()]
        taginfo.sort(key=lambda x: x[2], reverse=True)
        for tag, tag_counts, num_appearances in taginfo:
            num_different = len(tag_counts)
            names = [(name, count) for name, count in tag_counts.items()]
            names.sort(key=lambda x: x[1], reverse=True)
            lev1 = indent + lev0 + alen - len(str(num_appearances))
            if num_different == 1:
                print(' '*lev1+f"{num_appearances} {tag} = {''.join([k for k in tag_counts.keys()])}")
                continue
            if num_different > 10:  # usually only for the 'name' tag
                print(' '*lev1+f"{num_appearances} {tag} has {num_different} diff. values:")
                print(' '*(lev1+indent)+", ".join([f"({count}) {name}" for name, count in names]))
                continue
            print(' '*lev1+f"{num_appearances} {tag} =")
            for name, count in names:
                # right-align the counts
                clen = len(str(count))
                lev2 = indent + lev1 + alen - clen
                print(' '*lev2+f"{count} {name}")

DEFAULT_SUBSTATION_KV = 20.0  # OSM substations are rarely tagged with a clean voltage; this is
                               # a generic MV-distribution placeholder used whenever no usable
                               # 'voltage' tag is present. Same default is reused for standalone
                               # power=generator nodes (see docstring below for why they get a
                               # bus at all).
DEFAULT_PLANT_KV = 110.0       # power=plant nodes default to a sub-transmission interconnection
                               # voltage instead, since plants usually connect higher up than a
                               # neighborhood substation.


def _parse_voltage_kv(tags: dict, default_kv: float) -> float:
    """ OSM's 'voltage' tag is documented (OSM wiki) to be in volts, and may be a
    semicolon-separated list for a multi-voltage substation/line (take the max, mirroring
    add_transnet_bus's handling of transnet's own semicolon-separated voltage field in
    transnet_to_pp.py). Falls back to default_kv if the tag is missing or unparseable. """
    v = tags.get('voltage')
    if not v:
        return default_kv
    try:
        if ';' in v:
            v = max(float(x) for x in v.split(';') if x.strip())
        else:
            v = float(v)
        return v / 1000.0  # volts -> kV
    except (ValueError, TypeError):
        return default_kv


def _parse_power_mw(tag_value) -> float | None:
    """ parse an OSM power-output tag (e.g. generator:output:electricity='6 MW',
    plant:output:electricity='500 kW') into MW. Returns None if missing/unparseable, so callers
    can fall back to a default rather than mistaking an unparseable tag for a real 0 MW. """
    if not tag_value:
        return None
    m = re.match(r'^\s*([\d.]+)\s*([a-zA-Z]*)', str(tag_value))
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).upper()
    if unit.startswith('GW'):
        return val * 1000.0
    if unit.startswith('KW'):
        return val / 1000.0
    return val  # 'MW' or no unit given -> assume MW


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """ great-circle distance in km between two lat/lon points -- used to give OSM-derived
    lines a real length_km (OSM doesn't tag line length directly, but does give us both
    endpoints' coordinates via the substation/plant/generator nodes they connect). """
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def network_from_OSM(OSM_DATA) -> aux.pandapowerNet:
    """
    Create a pandapower network from OSM power=* data, following the same "buses first, then
    generators, then lines" pattern as transnet_to_pp() in transnet_to_pp.py.

    INPUT: OSM_DATA is whatever get_OSM_data() above returns -- either the (DATA, INDICES) or
    (DATA, INDICES, SUBTAGS) tuple -- or a bare DATA dict (INDICES is then rebuilt by scanning
    DATA for 'power' tags).

    This is a *basic, direct* conversion of what OSM explicitly gives us, NOT the full
    graph-inference engine described in the module TODO at the top of this file and in
    notebooks/OSM_power.ipynb's markdown cells (which is why transnet was used instead, for a
    more complete/curated network). Concretely, it:
     1. creates one bus per power=substation, power=plant, and power=generator *node* (way/
        relation substations are skipped -- see below)
     2. adds a pp.create_gen at each plant/generator bus
     3. adds a pp.create_line for every power=line/minor_line way whose OSM node path directly
        contains 2+ of the bus nodes from step 1, connecting consecutive touches along the path

    Known simplifications (intentionally left as-is, not "bugs"):
     - Only 'node'-type substations/plants/generators become buses. A substation mapped as a
       'way' (building outline) or 'relation' has no single lat/lon and, critically, no OSM
       *node* ID that a line's node-path could ever reference (way IDs and node IDs are
       independent ID spaces in OSM) -- so it can never be an explicit line endpoint anyway and
       is skipped rather than guessing a centroid for it.
     - A line way only connects to a bus when the line's own node path *directly* contains that
       bus's OSM node ID -- i.e. the line explicitly touches the substation/plant/generator
       node. It does NOT infer a connection when a line merely ends near (but not exactly at) a
       substation, e.g. at an untagged utility pole a few meters from the substation fence.
       That would need a spatial nearest-neighbor index (e.g. a k-d tree over all substation
       coordinates), which is out of scope for this basic pass.
     - power=generator nodes get their own bus even though that's not literally a "substation
       or plant" -- pp.create_gen needs *some* bus to attach to, and matching a standalone
       generator (e.g. a single wind turbine) to its nearest substation is exactly the
       nearest-neighbor inference ruled out above. This models each standalone generator as
       directly grid-connected at its own point, which is a simplification but doesn't
       fabricate connectivity the data doesn't contain.
     - Bus vn_kv is set once at creation time from the node's own 'voltage' tag (or a default)
       and is NOT reconciled against a connecting line's voltage tag by spinning up an offshoot
       bus + transformer the way transnet_to_pp.py's create_offshoot_bus does for multi-voltage
       substations -- OSM voltage tagging is too sparse/unreliable for that extra machinery to
       be worth it here.
     - OSM never tags conductor impedance, so unlike transnet_to_pp.py (which has real
       r_ohm_km/x_ohm_km/c_nf_km columns to build a std_type from), every OSM-derived line here
       just reuses pandapower's built-in "NAYY 4x150 SE" std_type as an inert placeholder --
       do not trust power-flow results computed from these line parameters.
    """
    net = pp.create_empty_network()

    # accept (DATA, INDICES[, SUBTAGS]) from get_OSM_data(), or a bare DATA dict
    if isinstance(OSM_DATA, (tuple, list)):
        DATA, INDICES = OSM_DATA[0], OSM_DATA[1]
    else:
        DATA = OSM_DATA
        INDICES = defaultdict(list)
        for id_, tags in DATA.items():
            if 'power' in tags:
                INDICES[tags['power']].append(id_)

    bus_of_node = {}  # OSM node id -> pandapower bus index (substation/plant/generator nodes only)

    # 1 + 2. buses (substation/plant/generator nodes) + generators (plant/generator nodes)
    for ptype in ('substation', 'plant', 'generator'):
        default_kv = DEFAULT_PLANT_KV if ptype == 'plant' else DEFAULT_SUBSTATION_KV
        for node_id in INDICES.get(ptype, []):
            tags = DATA[node_id]
            if tags.get('osm_type') != 'node':
                continue  # way/relation substations: no lat/lon, no line-referenceable node id
            try:
                lat, lon = float(tags['lat']), float(tags['lon'])
            except (KeyError, TypeError, ValueError):
                continue
            vn_kv = _parse_voltage_kv(tags, default_kv)
            name = tags.get('name', f'{ptype} {node_id}')
            bus_idx = pp.create_bus(net, vn_kv=vn_kv, name=name, geodata=(lat, lon), type='b')
            bus_of_node[node_id] = bus_idx

            if ptype in ('plant', 'generator'):
                p_mw = _parse_power_mw(tags.get('generator:output:electricity'))
                if p_mw is None:
                    p_mw = _parse_power_mw(tags.get('plant:output:electricity'))
                if p_mw is None:
                    # same defaults transnet_to_pp.py uses for its 'generator'/'plant' node types
                    p_mw = 50.0 if ptype == 'plant' else 10.0
                pp.create_gen(net, bus=bus_idx, p_mw=p_mw, vm_pu=1.0, name=name,
                               slack=False, type='unknown', controllable=True,
                               max_p_mw=p_mw, min_p_mw=0, max_q_mvar=0.01, min_q_mvar=0)

    # 3. lines: connect consecutive bus-touches along each line/minor_line way's node path
    for line_type in ('line', 'minor_line'):
        for way_id in INDICES.get(line_type, []):
            way = DATA[way_id]
            if way.get('osm_type') != 'way':
                continue
            touches = []  # [(node_id, bus_idx), ...] in path order along the way
            for ref in way.get('nodes', []):
                try:
                    ref_id = int(ref)
                except (TypeError, ValueError):
                    continue
                if ref_id in bus_of_node:
                    touches.append((ref_id, bus_of_node[ref_id]))
            if len(touches) < 2:
                # line doesn't directly touch >=2 known substation/plant/generator nodes --
                # skip it (see docstring: no nearest-neighbor inference here)
                continue
            vn_kv = _parse_voltage_kv(way, DEFAULT_SUBSTATION_KV)
            name = way.get('name', f'line {way_id}')
            for (n1, b1), (n2, b2) in zip(touches, touches[1:]):
                if b1 == b2:
                    continue
                (lat1, lon1), (lat2, lon2) = _bus_latlon(net, b1), _bus_latlon(net, b2)
                length_km = max(_haversine_km(lat1, lon1, lat2, lon2), 0.001)
                pp.create_line(net, from_bus=b1, to_bus=b2, length_km=length_km,
                                std_type="NAYY 4x150 SE", name=name, vn_kv=vn_kv)
    return net


def _bus_latlon(net: aux.pandapowerNet, bus_idx: int) -> tuple[float, float]:
    """ (lat, lon) of a bus. pandapower >=3.0 removed the bus_geodata table -- geodata is now a
    GeoJSON-Point string in net.bus['geo']; this repo's converters pass create_bus(geodata=(lat, lon))
    so the stored coordinates are [lat, lon]. """
    geo = net.bus.at[bus_idx, 'geo'] if 'geo' in net.bus.columns else None
    if isinstance(geo, str) and geo:
        try:
            c = json.loads(geo)['coordinates']
            return float(c[0]), float(c[1])
        except (ValueError, KeyError, IndexError):
            pass
    elif isinstance(geo, (list, tuple)) and len(geo) == 2:
        return float(geo[0]), float(geo[1])
    return 0.0, 0.0

if __name__ == '__main__':
    #bounding box coords around Ft. Lauderdale, FL, power plant
    bottom = 26.045381
    left = -80.219148
    top = 26.087833
    right = -80.169414

    ft_lauderdale = (left, bottom, right, top)

    #print('Getting data for Ft. Lauderdale power plant in Florida')
    #Df, If = get_OSM_power_data(ft_lauderdale)

    print('Getting data for Shiloh wind farm in California')
    shiloh_wind = (-121.9452,38.0780,-121.7295,38.2452)
    D, I, T = get_OSM_data(shiloh_wind, print_info=True)
    print_osm_info(I, T)
