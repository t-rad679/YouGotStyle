import tensor_io as tio
import style_transfer
import examples


def do_the_thing():
    """ Makeh dee artz """

    content_info = examples.femfam_info
    style_info = examples.simpsons_info

    content_image = tio.load_image_as_tensor(tio.fetch_file(*content_info))
    style_image = tio.load_image_as_tensor(tio.fetch_file(*style_info))

    stylized_image = style_transfer.style_transfer_easy_mode(content_image, style_image)
    final_image = tio.tensor_to_image(stylized_image)
    final_image.show()
    final_image.save(r'C:\Users\trevo\Pictures\simpsons_style_transfer.png')

