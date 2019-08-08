algebraic3d
left = Plane(0,0,0;-1,0,0);
bottom = Plane(0,0,0;0,-1,0);
back = Plane(0,0,0;0,0,-1);
right = Plane(1,0,0;1,0,0);
top = Plane(0,1,0;0,1,0);
front = Plane(0,0,1;0,0,1);
cube = left and bottom and back and right and top and front;
pore = Sphere(0.5,0.5,0.5;0.2);
cube_w_hole = cube and not pore;