import requests
import xml.etree.ElementTree as ET
import queue
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

def network_from_OSM(OSM_DATA) -> aux.pandapowerNet:
    """
    Create a pandapower network from the OSM data.
    1. infer network by looping over substations
    2. add lines between substations
    """
    net = pp.create_empty_network()

    raise NotImplementedError("Need to implement this OSM stuff")

    return net

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
