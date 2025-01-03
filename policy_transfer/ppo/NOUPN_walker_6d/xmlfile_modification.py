import xml.etree.ElementTree as ET
value = 3
tree = ET.parse('walker_1.xml')
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
y_cordinates = []
x_cordinates = []
#z_vals = []
body_count = 4                                      # Hopper Agent total parts = 4

for i in range(1,body_count+1):
    current_parent=parent_main+ ("body/" * i) +"/geom[@"
    parent_path = current_parent+child_id+"]"
    y,x,_ = find_val(root, parent_path, child_id)
    y_cordinates.append(y)
    x_cordinates.append(x)

# yl3 = 1.05      
# yl2 = 1.55     
yl_body1 = 1      
xl_body4 = 9.95

print(y_cordinates)
y_body0 = float(y_cordinates[3])
y_body1 = yl_body1 + y_body0           # y1-y0 gives length from base to y1 -> l_b3 (length of body3)
# y2 = y1 + (float(y_cordinates[1]) - float(y_cordinates[2]))
# y3 = y2 + (float(y_cordinates[0]) - float(y_cordinates[1]))
y_body2 = y_body1 + 0.45
y_body3 = y_body2 + 0.40
                               # y3 is the highest point in vertical axis   

                               # 0.1, 10.1, 11.15, 12.60

x_body0 = float(x_cordinates[3])

#setting horizontal part length values
if xl_body4!=None:
    xl = xl_body4 + x_body0
    x = root.find("./worldbody/body/body/body/body/geom[@fromto]")
    x_elem = x.attrib["fromto"]
    x1_list = x_elem.split()
    print("x1lst", x1_list)
    x1_list[3] = str(xl)
    x_temp = ' '.join(x1_list)
    x.attrib["fromto"] = x_temp

y_cordinate_vals = [y_body0, y_body1, y_body2, y_body3]
y_cordinate_vals.reverse()
print("new_y", y_cordinate_vals)

#setting vertical-part length values in xml
def set_val(root, parent_path, child_id, start_y, end_y=None):
    for parent in root.findall(parent_path):
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

for i in range(0,len(y_cordinate_vals)):
    j = i + 1
    start = y_cordinate_vals[i] 
    current_parent=parent_main+ ("body/" * j) +"/geom[@"
    parent_path = current_parent+child_id+"]"
    if j < 4:
        end = y_cordinate_vals[i+1]
        set_val(root, parent_path, child_id, start, end)
    else:
        set_val(root, parent_path, child_id, start)     #end is the same as start in this case (i.e. fixed point in groud)

tree.write("walker_out.xml")


# {'l_b1': 1.8496803846183816, 
# 'l_b2': 1.7675655557522811, 
# 'l_b3': 0.3456698463447919, 
# 'l_b4': 0.7930097062615558, 
# 't1': 0.04028065676012103, 
# 't2': 0.09661156911699899, 
# 't3': 0.10959988373221043, 
# 't4': 0.1754950721870359}
