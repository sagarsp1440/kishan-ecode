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

class Hopper:
    def __init__(self, name):
        self.name = name
    def construct_xml(self, params, xml_path):
        print("Constructing xml")
        # l_b1 = round(params.get('l_b1'), 4)
        # l_b2 = round(params.get('l_b2'), 4)
        l_b3 = round(params.get('l_b3'), 4)
        l_b4 = round(params.get('l_b4'), 4)
        # t3 = round(params.get('t3'), 4)
        # t4 = round(params.get('t4'), 4)
        t3 = 0.04
        t4 = 0.06
        
        ########### Xml file modification part - start ###############

        tree = ET.parse(xml_path)
        root = tree.getroot()

        def find_val(root, parent_path, child_id):
            parent = root.find(parent_path)
            body= parent.attrib[child_id]
            body_list= body.split()
            y = body_list[2]                      #height or length along vertical axis
            z = body_list[1]                      #width
            x = body_list[0]                      #length along horizonatal
            return y,x,z

        parent_main = "./worldbody/"
        child_id="fromto"
        y_vals = []
        x_vals = []
        z_vals = []

        body_count = 4                                      # Hopper Agent total parts = 4

        for i in range(1,body_count+1):
            current_parent=parent_main+ ("body/" * i) +"/geom[@"
            parent_path = current_parent+child_id+"]"
            y,x,_ = find_val(root, parent_path, child_id)
            y_vals.append(y)
            x_vals.append(x)

        # yl3 = 1.05      
        # yl2 = 1.55     
        yl1 = l_b3      
        xl4 = l_b4

        print(y_vals)
        y0 = float(y_vals[3])
        y1 = yl1 + y0           # y1-y0 gives length from base to y1 -> l_b3 (length of body3)
        # y2 = y1 + (float(y_vals[1]) - float(y_vals[2]))
        # y3 = y2 + (float(y_vals[0]) - float(y_vals[1]))
        y2 = y1 + 0.45
        y3 = y2 + 0.40
                                    # y3 is the highest point in vertical axis   

                                    # 0.1, 10.1, 11.15, 12.60

        x0 = float(x_vals[3])
        print("x", x)
        #setting horizontal part length values
        if xl4!=None:
            xl = xl4 + x0
            x = root.find("./worldbody/body/body/body/body/geom[@fromto]")
            x_elem = x.attrib["fromto"]
            x1_list = x_elem.split()
            print("x1lst", x1_list)
            x1_list[3] = str(xl)
            x_temp = ' '.join(x1_list)
            x.attrib["fromto"] = x_temp

        new_y = [y0, y1, y2, y3]
        new_y.reverse()
        print("new_y", new_y)

        #setting vertical-part length values in xml
        def set_val(root, parent_path, child_id, start_y, end_y=None):
            parent = root.find(parent_path)
            body= parent.attrib[child_id]
            body_list= body.split()
            #print("setting value", start_y)
            if child_id=="fromto":
                body_list[2] = str(start_y)
                if end_y!=None:
                    body_list[5] = str(end_y)

            temp = ' '.join(body_list)
            parent.attrib[child_id] = temp
            
        parent_main = "./worldbody/"
        child_id="fromto"
        y_vals = []

        for i in range(0,len(new_y)):
            j = i + 1
            start = new_y[i] 
            current_parent=parent_main+ ("body/" * j) +"/geom[@"
            parent_path = current_parent+child_id+"]"
            if j < 3:
                end = new_y[i+1]
                set_val(root, parent_path, child_id, start, end)
            else:
                set_val(root, parent_path, child_id, start)     #end is the same as start in this case (i.e. fixed point in groud)


        #Thickness values -> t3, t4
        # t_b3 = root.find("./worldbody/body/body/body/geom[@size]")
        # t_b3.attrib["size"] = str(t3)
        # t_b4 = root.find("./worldbody/body/body/body/body/geom[@size]")
        # t_b4.attrib["size"] = str(t4)

        print("New Hopper Ready!!!")
        tree.write(xml_path)

        new_params = [l_b3, l_b4]

        return new_params


