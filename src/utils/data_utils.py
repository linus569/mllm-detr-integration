from matplotlib import pyplot as plt


def show_img(img, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


def show_img_with_bbox(sample, dataset, title=None, figsize=(10, 10)):
    id_to_cat_name = dataset.index_to_cat_name
    images, bboxes, categories = (
        sample["images"],
        sample["instance_bboxes"],
        sample["instance_classes_id"],
    )

    batch_size = len(images)
    fig, axes = plt.subplots(
        1, batch_size, figsize=(figsize[0] * batch_size, figsize[1])
    )

    # Convert to list of axes if batch_size is 1
    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        img = images[i]
        bboxes1 = bboxes[i]
        categories1 = categories[i]

        img = img.permute(1, 2, 0).numpy()
        img = img - img.min()
        img = img / img.max()
        axes[i].imshow(img)
        axes[i].axis("off")

        if title is not None:
            axes[i].set_title(f"{title} {i+1}")

        for cat, bbox in zip(categories1, bboxes1):
            x1, y1, x2, y2 = bbox  # x_min, y_min, x_max, y_max -> YOLO format
            x1, y1, x2, y2 = (
                x1 * img.shape[1],
                y1 * img.shape[0],
                x2 * img.shape[1],
                y2 * img.shape[0],
            )
            print(x1, y1, x2, y2)
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
            )
            axes[i].add_patch(rect)

            # add label text to rect
            class_name = id_to_cat_name[cat.item()]
            axes[i].text(x1, y1 - 5, class_name, fontsize=12, color="red")

    plt.tight_layout()
    return axes
