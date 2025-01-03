import xml.etree.ElementTree as ET
import subprocess


def run(args2):
    args2 = [str(x) for x in args2]
    print("Run", args2)

    try:
        subprocess.check_call(args2)
        return 0
    except subprocess.CalledProcessError as e:
        print(e)
        return e.returncode

class AntEnv:
    def __init__(self, name):
        self.name = name
    def construct_xml(self, params, xml_path):
        
        print("Constructing xml")
        
        fll = round(params.get('fll'), 3)
        lfll = round(params.get('lfll'), 3)
        frl = round(params.get('frl'), 3)
        lfrl = round(params.get('lfrl'), 3)
        bll = round(params.get('bll'), 3)
        lbll = round(params.get('lbll'), 3)
        brl = round(params.get('brl'), 3)
        lbrl = round(params.get('lbrl'), 3)
    
        ########### Xml file modification part - start ###############

        tree = ET.parse(xml_path)
        root = tree.getroot()

        worldbody = root[5]
        torso = worldbody[2]

        bodypart={}
        body_list = []

        leg_pos_list = {'front_left_leg':'0.0 0.0 0.0'+ " " + str(fll)+ " " + str(fll)+ " " + '0.0',
                        'front_right_leg':'0.0 0.0 0.0'+ " " + str(frl)+ " " + str(-frl)+ " " + '0.0',
                        'back_leg': '0.0 0.0 0.0'+ " " + str(bll)+ " " + str(bll)+ " " + '0.0',
                        'right_back_leg':'0.0 0.0 0.0'+ " " + str(brl)+ " " + str(-brl)+ " " + '0.0'}

        lower_leg_pos_list = {'front_left_leg':'0.0 0.0 0.0'+ " " + str(lfll)+ " " + str(lfll)+ " " + '0.0',
                        'front_right_leg':'0.0 0.0 0.0'+ " " + str(lfrl)+ " " + str(-lfrl)+ " " + '0.0',
                        'back_leg': '0.0 0.0 0.0'+ " " + str(lbll)+ " " + str(lbll)+ " " + '0.0',
                        'right_back_leg':'0.0 0.0 0.0'+ " " + str(lbrl)+ " " + str(-lbrl)+ " " + '0.0'}


        for child_body in torso.findall('body'):
            #bodypart[body.attrib['name']] = body
            child1_geom_pos = child_body[0].attrib
            child1_geom_pos['fromto'] = leg_pos_list[child_body.attrib['name']]
            print("new",child1_geom_pos['fromto'])
            # child2_body_pos = child_body[1].attrib

            nxt_body = child_body[1]
            leg_aux_pos = child_body[1].attrib
            pos = leg_pos_list[child_body.attrib['name']].split(" ")
            leg_aux_pos['pos'] = str(pos[3]+" "+pos[4]+" "+pos[5])
            # print(leg_aux_pos)

            lower_leg_body_geom = child_body[1][2][1].attrib
            # print(lower_leg_body_geom["fromto"])
            lower_leg_body_geom["fromto"]=lower_leg_pos_list[child_body.attrib['name']]
            # print(lower_leg_body_geom)

        tree.write(xml_path)

        ########### Xml file modification part - end ###############

        new_params = [fll, lfll, frl, lfrl, bll, lbll, brl, lbrl]

        return new_params


