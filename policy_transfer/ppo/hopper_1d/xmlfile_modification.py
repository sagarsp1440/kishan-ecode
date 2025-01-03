import xml.etree.ElementTree as ET
abs_path = '/home/knagiredla/.conda/envs/py37rlzoo/lib/python3.7/site-packages/gym/envs/mujoco/assets/'
tree = ET.parse(abs_path+'hopper.xml')
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

yl1 = 1.05
yl2 = 1.55
yl3 = 1.6
xl4 = 1.95

y0 = float(y_vals[3])
y1 = yl1 + y0
y2 = yl2 + y1
y3 = yl3 + y2   

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
#print("new_y", new_y)

#setting vertical-parts length values in xml
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
    if i < 3:
        end = new_y[i+1]
        set_val(root, parent_path, child_id, start, end)
    else:
        set_val(root, parent_path, child_id, start)     #end is the same as start in this case (i.e. fixed point in groud)


#Thickness vales from HB
t1 = 0.02
t2 = 0.04
t3 = 0.05
t4 = 0.04

#Setting size values in xml
t_b1 = root.find("./worldbody/body/geom[@size]")
t_b1.attrib["size"] = str(t1)
t_b2 = root.find("./worldbody/body/body/geom[@size]")
t_b2.attrib["size"] = str(t2)
t_b3 = root.find("./worldbody/body/body/body/geom[@size]")
t_b3.attrib["size"] = str(t3)
t_b4 = root.find("./worldbody/body/body/body/body/geom[@size]")
t_b4.attrib["size"] = str(t4)

tree.write(abs_path+'hopper.xml')




