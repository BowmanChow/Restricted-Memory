
from glob import glob
from PIL import Image
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QMainWindow, QTextEdit, QVBoxLayout, QScrollArea, QHBoxLayout, QPushButton, QFileSystemModel, QListView, QSplitter, QLineEdit, QTableView, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, QPoint, QRect, QModelIndex, QSortFilterProxyModel
from PyQt5.QtGui import QPixmap, QPainter, QPen, QImage, QColor, QPalette, QWheelEvent
from PyQt5 import QtCore, QtGui
import sys
from utils.image import _palette
import pathlib
import random
import cv2
import numpy as np
import os
import torch

seq_name = "4014_cut_tomato"

result_root = "./results/aotplus_R50_AOTL/pre_vost/eval/vost/debug"
result_seq_path = os.path.join(result_root, seq_name)

image_root = "./datasets/VOST/JPEGImages_10fps"
image_seq_path = os.path.join(image_root, seq_name)

label_root = "./datasets/VOST/Annotations"
label_seq_path = os.path.join(label_root, seq_name)

all_image_files = os.listdir(image_seq_path)
all_image_files.sort()
img_name = all_image_files[20]
print(f"{img_name = }")

sample_loaded_tensor = torch.load(os.path.join(
    result_seq_path, f"{pathlib.Path(img_name).stem}_layer_mem.pt"))
ori_height, ori_width = sample_loaded_tensor["ori_height"], sample_loaded_tensor["ori_width"]
print(f"{ori_height = }  {ori_width = }")

down_h, down_w = int(ori_height.item()) // 2, int(ori_width.item()) // 2


def downsample_image(image) -> np.ndarray:
    return cv2.resize(image, (down_w, down_h), fx=0, fy=0, interpolation=cv2.INTER_AREA)


def put_text_on_image(image, text) -> np.ndarray:
    return cv2.putText(image, text, (image.shape[1]//2, image.shape[0]-1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(64, 64, 200), thickness=2)


def dot_on_image(image, position, color=(64, 64, 192)) -> np.ndarray:
    return cv2.circle(image, position, 3, color=color, thickness=3)


def read_concat_file_list(file_list):
    f_list = [cv2.imread(f) if os.path.exists(f) else np.zeros(
        (ori_height, ori_width, 3), dtype=np.uint8) for f in file_list]
    f_list = [downsample_image(i) for i in f_list]
    f_list = [put_text_on_image(image, pathlib.Path(
        file_list[i]).name) for i, image in enumerate(f_list)]
    f = np.concatenate(f_list, axis=1)
    return f


def map_indices(indices: tuple, origin_size: tuple, target_size: tuple):
    return_indices = np.array(indices, dtype=float) / np.array(
        origin_size, dtype=float) * np.array(target_size, dtype=float)
    return_indices = np.round(return_indices)
    return_indices = return_indices.astype(int)
    return tuple(return_indices)


palette_np = np.array(_palette).reshape((-1, 3))
h, w = sample_loaded_tensor["h"], sample_loaded_tensor["w"]
print(f"{h = }  {w = }")

del sample_loaded_tensor

def plot_per_image(point_list, per_layer_image: np.ndarray, attn_values, attn_indices, inner_pred_label, T_num):
    for pos_h, pos_w in point_list:
        print(f"{pos_h = }  {pos_w = }")
        # pos_h, pos_w = pos_h.item(), pos_w.item()
        pred_label_color = palette_np[inner_pred_label[pos_h, pos_w]]
        pred_label_color = [int(c) for c in pred_label_color]
        pred_label_color.reverse()
        start_point = map_indices((pos_w, pos_h), (w, h), (down_w, down_h))
        start_point = (start_point[0] + down_w * T_num, start_point[1])
        per_layer_image = dot_on_image(
            per_layer_image, start_point, color=pred_label_color)
        target_poses = attn_indices[pos_h, pos_w]
        target_pos_values = attn_values[pos_h, pos_w]
        # print(f"{target_poses = }")
        # print(f"{target_pos_values = }")
        # most_target_pos = torch.mode(target_pos).values
        target_pos_value_sum = 0
        for target_pos, target_pos_value in zip(target_poses, target_pos_values):
            if target_pos_value_sum > 0.25:
                # print(f"trunc on {target_pos_value = }")
                break
            target_pos_value_sum += target_pos_value
            # if target_pos_value < 0.01:
            #     break
            # print(f"{target_pos = }")
            # print(f"{target_pos_value = }")
            if len(target_pos) == 3:
                target_pos_T, target_pos_h, target_pos_w = tuple(target_pos)
            elif len(target_pos) == 2:
                target_pos_h, target_pos_w = tuple(target_pos)
                target_pos_T = 0
            end_point = map_indices((target_pos_w, target_pos_h), (w, h), (down_w, down_h))
            end_point = (end_point[0] + target_pos_T * down_w, end_point[1])
            per_layer_image_overlay = per_layer_image.copy()
            per_layer_image_overlay = cv2.line(
                per_layer_image_overlay, start_point, end_point, color=pred_label_color, thickness=2, lineType=cv2.LINE_AA)
            per_layer_image_overlay = dot_on_image(
                per_layer_image_overlay, end_point, color=pred_label_color)
            line_alpha = min(target_pos_value / 0.25, 1)
            per_layer_image = cv2.addWeighted(
                per_layer_image_overlay, line_alpha, per_layer_image, 1-line_alpha, 0)
    return per_layer_image


def draw_point_on_image(point_list, attn_weights, image_list, inner_pred_label, T_num):
    concat_images = []
    for i, (attn_weight, image) in enumerate(zip(attn_weights, image_list)):
        print(f"Layer {i}: ")
        plot_image = plot_per_image(
            point_list,
            image,
            attn_weight["attn_values"],
            attn_weight["attn_indices"],
            inner_pred_label,
            T_num,
        )
        # print(f"{plot_image.shape = }  {plot_image.dtype = }")
        concat_images.append(plot_image)
    if len(concat_images) < len(image_list):
        concat_images.extend(image_list[len(concat_images):])
    return concat_images


def load_image_pred_gt_from_file(image_file_list):
    image = read_concat_file_list(
        [os.path.join(image_seq_path, i) for i in image_file_list])
    pred_mask = read_concat_file_list([os.path.join(
        result_seq_path, pathlib.Path(i).with_suffix(".png")) for i in image_file_list])
    # print(f"{pred_mask.shape = }  {pred_mask.dtype = }")
    gt_mask = read_concat_file_list([os.path.join(
        label_seq_path, pathlib.Path(i).with_suffix(".png")) for i in image_file_list])
    # print(f"{gt_mask.shape = }  {gt_mask.dtype = }")
    # concat_images = np.concatenate(concat_images, axis=0)
    return image, pred_mask, gt_mask


def load_mem_from_file(image_file, is_long=True):
    print(f"\nLoading from memory {image_file}")
    loaded_tensor = torch.load(os.path.join(
        result_seq_path, f"{pathlib.Path(image_file).stem}_layer_mem.pt"))
    frame_index = all_image_files.index(image_file)
    if is_long:
        record_T = loaded_tensor["long_mem_len"]
        memory_indices = loaded_tensor["memory_indices"]
        print(f"{memory_indices = }")
        memory_img_files = np.array(all_image_files)[memory_indices]
        memory_img_files = list(memory_img_files)
        memory_img_files = memory_img_files + [image_file]
        # memory_img_files = [all_image_files[0]] + memory_img_files
    else:
        record_T = 1
        memory_img_files = all_image_files[frame_index-1:frame_index+1]
    print(f"{record_T = }")
    print(f"{memory_img_files = }")
    inner_pred_label = loaded_tensor["inner_pred_label"]
    print(f"{inner_pred_label.shape = }")
    attn_weight = loaded_tensor["attn_weights"] if is_long else loaded_tensor["short_attn_weights"]
    return memory_img_files, record_T, inner_pred_label, attn_weight


class ImageScene(QGraphicsScene):
    def __init__(self, parent=None, is_long=True):
        super().__init__(parent)
        # self.points = []
        self._pixmap_item = QGraphicsPixmapItem()
        self.addItem(self._pixmap_item)
        self.setSceneRect(0, 0, 500, 500)
        self.is_long = is_long
        # self.show_image_and_points()
        self._other_mouse_function = []

    def show_image_and_points(self, points=[]):
        points_ = [(p.y() % down_h, p.x() % down_w) for p in points]
        self.image_list = draw_point_on_image(
            [map_indices(p, (down_h, down_w), (h, w)) for p in points_],
            self.attn_weights,
            self.image_list,
            self.inner_pred_label,
            self.record_T if self.is_long else 1,
        )
        image = np.concatenate(
            self.image_list + [self.gt_mask], axis=0)
        image = QImage(
            image.data, image.shape[1], image.shape[0], 3*image.shape[1], QImage.Format_BGR888)
        self._pixmap_item.setPixmap(QPixmap(image))
        self.setSceneRect(0, 0, image.width(), image.height())
        print(f"Set Image !!")

    def mouse_click(self, ev):
        if ev.button() != Qt.RightButton:
            return
        x = ev.scenePos().x()
        y = ev.scenePos().y()
        print(f"click on position : {x}, {y}")
        # self.points.append(ev.scenePos())
        self.show_image_and_points([ev.scenePos()])

    def mousePressEvent(self, ev) -> None:
        self.mouse_click(ev)
        for func in self._other_mouse_function:
            func(ev)

    def set_image(self, image_file: str):
        memory_img_files, self.record_T, self.inner_pred_label, self.attn_weights = load_mem_from_file(
            image_file, self.is_long)
        self.image, self.pred_mask, self.gt_mask = load_image_pred_gt_from_file(
            memory_img_files)
        self.image_list = [self.image.copy(), self.image.copy(),
                           self.image.copy(), self.pred_mask.copy(),]
        self.clear_current_input()

    def clear_current_input(self):
        # self.points.clear()
        self.image_list = [self.image.copy(), self.image.copy(),
                           self.image.copy(), self.pred_mask.copy(),]
        self.show_image_and_points()

    def connect_mouse_function(self, func):
        self._other_mouse_function.append(func)

    def save_image(self, file_name: str):
        image = np.concatenate(
            self.image_list + [self.gt_mask], axis=0)
        cv2.imwrite(f"{file_name}.png", image)


class ImageViewer(QGraphicsView):
    factor = 1.25

    def __init__(self, parent=None, is_long=True):
        super().__init__(parent)
        self.setRenderHints(
            QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        # self.setBackgroundRole(QPalette.Dark)

        self.scene = ImageScene(self, is_long=is_long)
        self.setScene(self.scene)

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self._other_scale_functions = []

    def zoomIn(self):
        self.zoom(self.factor)

    def zoomOut(self):
        self.zoom(1 / self.factor)

    def zoom(self, f):
        self.scale(f, f)

    def inner_scale(self, sx: float, sy: float) -> None:
        return super().scale(sx, sy)

    def scale(self, sx: float, sy: float) -> None:
        self.inner_scale(sx, sy)
        for func in self._other_scale_functions:
            func(sx, sy)
        return

    def connect_scale_function(self, func):
        self._other_scale_functions.append(func)

    def wheelEvent(self, event: QWheelEvent) -> None:
        modifiers = QApplication.keyboardModifiers()
        if modifiers != Qt.ControlModifier:
            return super().wheelEvent(event)
        if event.angleDelta().y() > 0:
            self.zoomIn()
        else:
            self.zoomOut()
        return

    def resetZoom(self):
        self.resetTransform()

    def fitToWindow(self):
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        centeral_widget = QWidget(self)
        layout = QHBoxLayout()
        centeral_widget.setLayout(layout)
        self.setCentralWidget(centeral_widget)

        h_splitter = QSplitter(Qt.Horizontal)

        image_layout = QVBoxLayout()

        self.label_text = QLineEdit()
        image_layout.addWidget(self.label_text)

        # self.img_label = ImageLabel(self)
        self.img_label = ImageViewer(self)
        self.img_label_short = ImageViewer(self, is_long=False)

        self.img_label.scene.connect_mouse_function(
            self.img_label_short.scene.mouse_click)
        self.img_label_short.scene.connect_mouse_function(
            self.img_label.scene.mouse_click)

        self.img_label.connect_scale_function(self.img_label_short.inner_scale)
        self.img_label_short.connect_scale_function(self.img_label.inner_scale)
        # scroll_area = QScrollArea()
        # scroll_area.setWidget(self.img_label)

        image_layout.addWidget(self.img_label)

        buttons_layout = QHBoxLayout()

        clear_current_button = QPushButton("clear current")
        clear_current_button.clicked.connect(
            self.img_label.scene.clear_current_input)
        clear_current_button.clicked.connect(
            self.img_label_short.scene.clear_current_input)
        buttons_layout.addWidget(clear_current_button)

        save_button = QPushButton("save")
        save_button.clicked.connect(
            lambda: self.img_label.scene.save_image(self.label_text.text()))
        buttons_layout.addWidget(save_button)

        image_layout.addLayout(buttons_layout)

        image_widget = QWidget()
        image_widget.setLayout(image_layout)
        h_splitter.addWidget(image_widget)

        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(image_seq_path)
        self.file_list_view = QListView()
        self.file_list_view.setModel(self.file_model)
        self.file_list_view.setRootIndex(self.file_model.index(image_seq_path))
        self.file_list_view.selectionModel().selectionChanged.connect(
            self.file_selected
        )

        h_splitter.addWidget(self.file_list_view)

        short_image_layout = QVBoxLayout()

        self.short_label_text = QLineEdit()
        short_image_layout.addWidget(self.short_label_text)

        short_image_layout.addWidget(self.img_label_short)

        short_save_button = QPushButton("save")
        short_save_button.clicked.connect(
            lambda: self.img_label_short.scene.save_image(self.short_label_text.text()))
        short_image_layout.addWidget(short_save_button)

        short_image_widget = QWidget()
        short_image_widget.setLayout(short_image_layout)
        h_splitter.addWidget(short_image_widget)

        layout.addWidget(h_splitter)

    def file_selected(self):
        for ix in self.file_list_view.selectedIndexes():
            print(f"List select {ix.row() = }  {ix.column() = }")
            value = ix.sibling(ix.row(), ix.column()).data()
            print(f"List select {value = }")
            # file_path = os.path.join(self.file_model.rootPath(), value)
            # self.img_label.set_image(value)
            self.img_label.scene.set_image(value)
            self.img_label_short.scene.set_image(value)
            return


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
