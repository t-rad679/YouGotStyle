from tensor_io import fetch_file, load_image_as_tensor


# Content examples
dawg_info = (
        'dawg.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg'
)
dawg_path = fetch_file(*dawg_info)
dawg_image = load_image_as_tensor(dawg_path)

sunset_info = (
        'sunset.jpg',
        'https://assets.simpleviewinc.com/simpleview/image/upload/c_fill,h_575,q_60,w_1024/v1/clients/asheville/171204_cvb_parkway_007_087513da-71ce-488c-bad2-abb5e4a3bd92.jpg'
)
# sunset_path = fetch_file(*sunset_info)
sunset_path = 'C:\\Users\\trevo\\Pictures\\projects\\project_valentine\\sunset.jpg'
sunset_image = load_image_as_tensor(sunset_path)
# femfam_info = (
#         'us.jpg',
#         'https://lh3.googleusercontent.com/NIsLjDOqxmRHoGEiqdnlKcvgnjKgENW5C3DemDOImSm4GMqp_tVSWiUO6Sem90MSoOwjcxTBd3PHTSXr6wa4InMsp8FE2YtYwfDUr6VCW72bp5eqOrEq8wL5WGDxmJM5wTx50i7cOZqFhDmz8QZRyh1WpRTIHCKJC24zO6ptdgLePZ3B9y4ZJmRz-jNX8TNAw-x8lhZKO7mbCMKcDFbko9oWimm3bE6P32JDT68QUqVyQI6RUClni5p9uaVanmY_XJPObS-2IV-4E_4zTobZE4ZIlf44MgCkr3WIWXTr_3eMYnVNybegbbTGEIa3_ageCGMeS7ONjAk9RJ-gb2VSYeNlveBHC9q4-zogWdyPcEcGlCZA_XTyqozgGudINGZ4GZimXkXaB9eabGGW2W6shrzjqECrLGCOF7ONRGki7fo9GYoyMUTD5QI-9VmdcxwyGPyyL2k7iM_qevtaHNSmFkTRt59NWhfXzqwOpD3iPZrdJibJQUAGcleqT81wTnOk7cTMrFv04s6gYigcUr0f3IOvLX5sUNjdiPYKuC8LIPM3ECt4YMFsqraOz1jG3TshPLOKPTYt2i47lqJIGkOhw8kgjdKmpvOYqhidjoXC9ulSfhLhfief9PGKS8s_bh5vOmENJrECaYAHqBpoLfe8UhY0GNsYus2ZC3qEw6qrXtyvOiEB-rDWAw=w958-h539-no'
# )
# femfam_path = fetch_file(*femfam_info)
# femfam_image = load_image_as_tensor(femfam_path)

# Style examples
simpsons_info = (
        'simpsons.jpg',
        'https://cl.goliath.com/image/upload/t_tn,f_auto,q_auto,$h_480,$w_895/go/2016/06/the-simpsons.png'
)
simpsons_path = fetch_file(*simpsons_info)
simpsons_image = load_image_as_tensor(simpsons_path)

kandinsky_info = (
        'fukken_weird.jpg',
        'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)
kandinsky_path = fetch_file(*kandinsky_info)
kandinsky_image = load_image_as_tensor(kandinsky_path)

style_sunset_info = (
        'pencil_sunset.jpg',
        'http://capappasart.com/wp-content/uploads/2013/02/birthday-sunset-96-685px-600x371.jpg'
)
# style_sunset_path = fetch_file(*style_sunset_info)
style_sunset_path = 'C:\\Users\\trevo\\Pictures\\projects\\project_valentine\\style_sunset.jpg'
style_sunset_image = load_image_as_tensor(style_sunset_path)
