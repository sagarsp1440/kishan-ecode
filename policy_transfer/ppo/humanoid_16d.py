import xml.etree.ElementTree as ET


class Humanoid16D:
    def __init__(self, name):
        self.name = name
    def construct_xml(self, params, xml_path):
        print("Constructing xml for Humanoid")
        # Default Humanoid Values
        # l_th = -0.34 
        # l_sh = -0.3
        # r_th = -0.34 
        # r_sh = -0.3 
        # r_s_s = 0.049
        # r_t_s = 0.06
        # l_s_s = 0.049
        # l_t_s = 0.06
        # l_f_s = 0.075
        # r_f_s = 0.075
        # l_u_a = .24
        # l_l_a = .17
        # l_h = 0.06 
        # r_u_a = .24 
        # r_l_a = .17
        # r_h = 0.06

        l_th = round(params.get('l_th'), 4)
        l_sh = round(params.get('l_sh'), 4)
        r_th = round(params.get('r_th'), 4)
        r_sh = round(params.get('r_sh'), 4)
        l_f_s = round(params.get('l_f_s'), 4)
        r_f_s = round(params.get('r_f_s'), 4)
        l_s_s = round(params.get('l_s_s'), 4)
        r_s_s = round(params.get('r_s_s'), 4)
        l_t_s = round(params.get('l_t_s'), 4)
        r_t_s = round(params.get('r_t_s'), 4)
        l_u_a = round(params.get('l_u_a'), 4)
        l_l_a = round(params.get('l_l_a'), 4)
        l_h = round(params.get('l_h'), 4)
        r_u_a = round(params.get('r_u_a'), 4)
        r_l_a = round(params.get('r_l_a'), 4)
        r_h = round(params.get('r_h'), 4)
    
        # ########### Xml file modification part - start ###############

        def compute_pos(value, initial_fromto, initial_pos):
            points_of_value = value.split()
            print("POV", points_of_value)
            points_of_initialfromto = initial_fromto.split()
            x1, y1, z1, x2, y2, z2 = map(float, points_of_value)
            x3, y3, z3, x4, y4, z4 = map(float, points_of_initialfromto)

            x_diff = x2 - x4
            y_diff = y2 - y4
            z_diff = z2 - z4

            points_of_pos = initial_pos.split()
            x5, y5, z5 = map(float, points_of_pos)
            x5 = x5 + x_diff
            y5 = y5 + y_diff
            z5 = z5 + z_diff

            return f"{x5} {y5} {z5}"

        tree = ET.parse(xml_path)
        root = tree.getroot()

        values_dict = {
            'left_thigh': '0 0 0 0 -0.01' + ' ' + str(l_th),
            'left_shin': '0 0 0 0 0' + ' '+ str(l_sh),
            'right_thigh': '0 0 0 0 0.01' + ' ' + str(r_th),
            'right_shin': '0 0 0 0 0' + ' ' + str(r_sh),
            'right_upper_arm': '0 0 0' + ' ' + str(r_u_a) + ' '+ str(-r_u_a) + ' '+ str(-r_u_a),
            'right_lower_arm': '0 0 0' + ' ' + str(r_l_a) + ' '+ str(r_l_a) + ' '+ str(r_l_a),
            'left_upper_arm': '0 0 0' + ' ' + str(l_u_a) + ' '+ str(l_u_a) + ' '+ str(-l_u_a),
            'left_lower_arm': '0 0 0' + ' ' + str(l_u_a) + ' '+ str(-l_u_a) + ' '+ str(l_u_a)
        }

        for key, new_fromto in values_dict.items():
            # Find all <body> elements with the specified name
            body_elements = root.findall(f".//body[@name='{key}']")

            geom = body_elements[0].find('geom')
            if geom is not None:
                initial_fromto = geom.get('fromto')
                # Update the "fromto" attribute value
                geom.set('fromto', new_fromto)

            if "thigh" in key:
                shin_body = body_elements[0].find("body")
                initial_pos = shin_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_pos)
                shin_body.set('pos', pos)

            if "shin" in key:
                foot_body = body_elements[0].find("body")
                initial_pos = foot_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_pos)
                foot_body.set('pos', pos)
            
            if "upper_arm" in key:
                arm_body = body_elements[0].find("body")
                initial_arm_pos = arm_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_arm_pos)
                arm_body.set('pos', pos)

            if "lower_arm" in key:
                arm_lower_body = body_elements[0].find("body")
                initial_arm_pos = arm_lower_body.get('pos')
                pos = compute_pos(new_fromto, initial_fromto, initial_arm_pos)
                arm_lower_body.set('pos', pos)

        size_dict = {
            'left_foot': str(l_f_s),
            'right_foot': str(r_f_s),
            'left_shin': str(l_s_s),
            'right_shin': str(r_s_s),
            'left_thigh': str(l_t_s), 
            'right_thigh': str(r_t_s),
            'left_hand': str(l_h),
            'right_hand': str(r_h)
        }
        # Iterate over the dictionary and update the size attribute for the corresponding bodies
        for key, value in size_dict.items():
            body = root.find(".//body[@name='" + key + "']")
            if body is not None:
                geom = body.find('geom')
                if geom is not None:
                    # Update the size attribute value
                    geom.set('size', value)
                    print(f"Updated the 'size' attribute value to '{value}' for <geom> in <body> with name='{key}'.")
                else:
                    print(f"<geom> element not found in the <body> element with name='{key}'.")
            else:
                print(f"No <body> element with name='{key}' found.")

        tree.write(xml_path)

        ########### Xml file modification part - end ###############

        print("New Humanoid Ready!!!")

        new_params = [l_th,l_sh,l_f_s,l_t_s,l_s_s,r_th,r_sh,r_f_s,r_t_s,r_s_s,l_u_a,l_l_a,l_h,r_u_a,r_l_a,r_h]

        return new_params



