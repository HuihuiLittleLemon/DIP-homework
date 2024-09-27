import cv2
import numpy as np
import torch
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行刚性变换，逐像素处理
# def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
#     """ 
#     Return
#     ------
#         A deformed image.
#     """
#     if source_pts.shape[0] != target_pts.shape[0]:
#         print("Odd input control points")
#         return
#     warped_image = np.zeros_like(image)
#     ### FILL: 基于MLS or RBF 实现 image warping
#     y, x = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
#     gfv = np.column_stack((x.reshape(-1), y.reshape(-1)))

#     for fv in gfv:
#         p_minus_v = target_pts - fv    
#         w = 1/(np.power(np.sum(p_minus_v**2, axis=1),alpha) + eps)
#         q_star = (w[:, np.newaxis]*target_pts).sum(axis=0)/w.sum()
#         q_hat = target_pts - q_star
#         p_star = (w[:, np.newaxis]*source_pts).sum(axis=0)/w.sum()
#         p_hat = source_pts - p_star
#         v_minus_q_star = fv - q_star
#         qhat_vert = np.column_stack((-q_hat[:, 1], q_hat[:, 0]))
#         v_minus_q_star_vert = np.column_stack((-v_minus_q_star[1], v_minus_q_star[0]))
#         Ai = w[:, np.newaxis, np.newaxis]*np.stack((q_hat, -qhat_vert), axis=1)@np.stack((v_minus_q_star.reshape(1,-1), -v_minus_q_star_vert), axis=1)
#         v_vec = (np.matmul(p_hat[:, np.newaxis, :], Ai).squeeze(1)).sum(axis=0)
#         v = np.linalg.norm(v_minus_q_star, axis=0)*v_vec/(np.linalg.norm(v_vec, axis=0) + eps) + p_star
#         v_int = np.floor(v).astype(int)
#         if np.any(np.floor(v_int)>=np.array([image.shape[1],image.shape[0]])) or np.any(np.floor(v_int)<0):
#             continue
#         else:        
#             warped_image[fv[1], fv[0], :] = image[v_int[1], v_int[0], :]

#     return warped_image

# 执行刚性变换，torch 并行
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    if source_pts.shape[0] != target_pts.shape[0] or source_pts.shape[0] == 1 :
        print("Odd input control points")
        return image
    # Convert input to torch tensors
    image_tensor = torch.from_numpy(image).int()
    source_pts_tensor = torch.from_numpy(source_pts).float()
    target_pts_tensor = torch.from_numpy(target_pts).float()

    # Initialize the warped image
    warped_image = torch.zeros_like(image_tensor, dtype=torch.int32)
    
    # Create grid of coordinates
    y, x = torch.meshgrid(torch.arange(image.shape[0]), torch.arange(image.shape[1]))
    gfv = torch.stack((x.ravel(), y.ravel()), dim=1).float()

    p_minus_v = target_pts_tensor.unsqueeze(1) - gfv.unsqueeze(0)  # [n_pts, 1, 2] - [1, num_pixels, 2]
    w = 1 / (torch.sum(p_minus_v**2, dim=2).pow(alpha) + eps)  # [n_pts, num_pixels]

    w_sum = w.sum(dim=0)
    q_star = (w.unsqueeze(2) * target_pts_tensor.unsqueeze(1)).sum(dim=0) / w_sum.unsqueeze(1)
    p_star = (w.unsqueeze(2) * source_pts_tensor.unsqueeze(1)).sum(dim=0) / w_sum.unsqueeze(1)

    q_hat = target_pts_tensor.unsqueeze(1) - q_star  # [n_pts, num_pixels, 2]
    p_hat = source_pts_tensor.unsqueeze(1) - p_star  # [n_pts, num_pixels, 2]
    v_minus_q_star = gfv - q_star  # [num_pixels, 2]

    qhat_vert = torch.stack((-q_hat[..., 1], q_hat[..., 0]), dim=-1)  # [n_pts, num_pixels, 2]
    v_minus_q_star_vert = torch.stack((-v_minus_q_star[..., 1], v_minus_q_star[..., 0]), dim=-1)  # [num_pixels, 2]

    v_minus_q_star_combine = torch.cat((v_minus_q_star.unsqueeze(1), -v_minus_q_star_vert.unsqueeze(1)), dim=1) # q_hat_combine: [num_pixels, 2, 2],q_hat_combine每个[2,2]维子矩阵都是对称矩阵
    q_hat_combine = torch.cat([q_hat.unsqueeze(2), -qhat_vert.unsqueeze(2)], dim=2) # q_hat_combine: [n_pts, num_pixels, 2, 2]
    Ai = w.unsqueeze(2).unsqueeze(3) * q_hat_combine@v_minus_q_star_combine.unsqueeze(0)  # Ai: [n_pts, num_pixels, 2, 2]
    
    # v_minus_q_star @ Ai,当Ai在前面，需要转置一下
    v_vec = torch.matmul(Ai.permute(0,1,3,2), p_hat.unsqueeze(-1)).squeeze(-1).sum(dim=0)  # [num_pixels, 2]
    v = torch.norm(v_minus_q_star, dim=1, keepdim=True) * v_vec / (torch.norm(v_vec, dim=1, keepdim=True) + eps) + p_star
    v_int = torch.floor(v).long()  # [num_pixels, 2]
    valid_mask = ((v_int[:, 0] >= 0) & (v_int[:, 0] < image.shape[1]) &
                (v_int[:, 1] >= 0) & (v_int[:, 1] < image.shape[0]))

    # Using the valid mask to get valid indices
    v_int_valid = v_int[valid_mask]  # valid pixel indices
    gfv_valid = gfv[valid_mask]  # valid grid points

    # Assign warped image values
    # Ensure indexing does not go out of bounds
    if len(v_int_valid) > 0:
        warped_image[gfv_valid[:, 1].long(), gfv_valid[:, 0].long()] = \
            image_tensor[v_int_valid[:, 1].long(), v_int_valid[:, 0].long()]

    return warped_image.numpy()

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
