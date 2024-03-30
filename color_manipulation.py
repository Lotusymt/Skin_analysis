import cv2
import numpy as np
import pandas as pd
import xlsxwriter
#takes one image and create a redness enhanced version
class ConvertToRedness:
    def __init__(self, image):
        self.image_org = image
        self.image_preprocessed = None
        self.image_lab = None
        self.result = None
        self.result_lab = None
        self.l_channel = None
        self.a_channel = None
        self.b_channel = None

    # Image preprocessing
    def preprocess_image(self):
        self.image_lab = cv2.cvtColor(self.image_org, cv2.COLOR_BGR2Lab)

        l,a,b = cv2.split(self.image_lab)

        # self.l_channel = cv2.equalizeHist(l)
        # self.a_channel = cv2.equalizeHist(a)
        # #self.b_channel = cv2.equalizeHist(b)
        # self.b_channel = b

        self.image_preprocessed = self.image_org
        #self.image_preprocessed = cv2.merge([self.l_channel, self.a_channel, self.b_channel])
        #ensure proper visualization
        #self.image_preprocessed = cv2.cvtColor(self.image_preprocessed, cv2.COLOR_Lab2BGR)
    # Visualization and interpretation
    def visualize_redness(self):
        # Define if the pixel is red or not
        # on the ab plane, if within a radius of r from red extreme, then it is red
        r = 115
        self.image_preprocessed= cv2.cvtColor(self.image_preprocessed, cv2.COLOR_BGR2Lab)
        # turn the channels into normal flexible type
        self.l_channel,self.a_channel,self.b_channel = cv2.split(self.image_preprocessed)
        self.image_preprocessed = cv2.cvtColor(self.image_preprocessed, cv2.COLOR_Lab2BGR)
        self.a_channel = self.a_channel.astype(np.int32)
        self.l_channel = self.l_channel.astype(np.int32)
        self.b_channel = self.b_channel.astype(np.int32)
        a_ = self.a_channel-255
        b_ = self.b_channel-128
        a = np.sqrt(np.square(a_) + np.square(b_))
        print('a min: ', np.min(a))

        print('a max: ', np.max(a))
        red_mask = a <= r
        rest_mask = a > r
        # create another image making the pixels of the red_mask black and the remaining white
        redness_image = np.zeros_like(self.image_preprocessed)
        redness_image[red_mask] = self.image_preprocessed[red_mask]
        redness_image[rest_mask] = 255
        # Save the image
        save_path = './test_result/redness_image.jpg'
        cv2.imwrite(save_path, redness_image)

        # gather range info in three channels
        # Gather the red pixels in each channel
        red_pixels_l = self.l_channel[red_mask]
        red_pixels_a = self.a_channel[red_mask]
        #red_pixels_b = self.b_channel[red_mask]

        # Calculate the extreme values f red pixels in each channel
        l_min, l_max = np.min(red_pixels_l), np.max(red_pixels_l)
        a_min, a_max = np.min(red_pixels_a), np.max(red_pixels_a)
        #b_min, b_max = np.min(red_pixels_b), np.max(red_pixels_b)

        # histogram equalization on l, a and b in each channel

        # lower bound of l is l_min - 20% of the range if it is positive, otherwise 0
        lower_bound_l_equ = l_min - (l_max - l_min) * 0.2 if l_min - (l_max-l_min)*0.2 > 0 else 0
        # upper bound of l is l_max - 20% of the range if it is less than 255, otherwise 255
        upper_bound_l_equ = l_max - (l_max - l_min) * 0.2 if l_max + (l_max-l_min)*0.2 < 255 else 255

        self.l_channel[red_mask] = (self.l_channel[red_mask] - l_min) * ((upper_bound_l_equ - lower_bound_l_equ)/(l_max - l_min)) + lower_bound_l_equ
        # # lower bound of b is the half of it
        # lower_bound_b_equ = b_min*0.5
        # # upper bound of b is the half of it
        # upper_bound_b_equ = b_max * 0.5
        # self.b_channel[red_mask] = (self.b_channel[red_mask] - b_min) * ((upper_bound_b_equ - lower_bound_b_equ)/(b_max - b_min)) + lower_bound_b_equ
        # lower bound of a is a_min + 20% of the range
        lower_bound_a_equ = a_min + (a_max - a_min) * 0.5
        # upper bound of a is a_max + 20% of the range
        upper_bound_a_equ = a_max + (a_max - a_min) * 1
        self.a_channel[red_mask] = (self.a_channel[red_mask] - a_min) * ((upper_bound_a_equ - lower_bound_a_equ)/(a_max - a_min)) + lower_bound_a_equ

        # for the rest pixels, increase l by 20% and move the point on the ab plane towards the origin by 50%
        self.l_channel[rest_mask] = np.where(
            self.l_channel[rest_mask] * 1.3 < 255,
            self.l_channel[rest_mask] * 1.3,
            255
        )
        self.a_channel[rest_mask] = np.full_like(self.a_channel[rest_mask], 127)
        self.b_channel[rest_mask] = np.full_like(self.b_channel[rest_mask], 127)
        #self.a_channel[rest_mask] = self.a_channel[rest_mask] * 0.5
        #self.b_channel[rest_mask] = self.b_channel[rest_mask] * 0.5

        # turn the channels back to uint8
        self.a_channel = self.a_channel.astype(np.uint8)
        self.l_channel = self.l_channel.astype(np.uint8)
        #turn channel b to all 127
        self.b_channel = np.full_like(self.b_channel, 127).astype(np.uint8)

        if self.l_channel.shape == self.a_channel.shape == self.b_channel.shape:
            # Check if the channels have the same depth (data type)
            if self.l_channel.dtype == self.a_channel.dtype == self.b_channel.dtype:
                # Merge the channels
                self.result_lab = cv2.merge([self.l_channel, self.a_channel, self.b_channel])
            else:
                print("Channel depths (data types) do not match.")
        else:
            print("Channel dimensions do not match.")
        #modified_cielab = cv2.merge([self.l_channel, self.a_channel, self.b_channel])

        # Convert the modified CIELab image back to the original color space
        self.result = cv2.cvtColor(self.result_lab, cv2.COLOR_Lab2RGB)
        self.result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        # Display the result
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_path = './test_result/sample_R.jpg'
        cv2.imwrite(save_path, self.result)
        save_path = './test_result/image_preprocessed.jpg'
        cv2.imwrite(save_path, self.image_preprocessed)
        save_path = './test_result/image_org.jpg'
        cv2.imwrite(save_path, self.image_org)
        #save_path = './test_result/image_lab.jpg'
        #cv2.imwrite(save_path, self.image_lab)
    def visualize_y(self):
        # Define if the pixel is red or not
        # on the ab plane, if within a radius of r from red extreme, then it is red
        r = 77
        self.image_preprocessed = cv2.cvtColor(self.image_preprocessed, cv2.COLOR_BGR2Lab)
        # turn the channels into normal flexible type
        self.l_channel, self.a_channel, self.b_channel = cv2.split(self.image_preprocessed)
        self.image_preprocessed = cv2.cvtColor(self.image_preprocessed, cv2.COLOR_Lab2BGR)
        self.a_channel = self.a_channel.astype(np.int32)
        self.l_channel = self.l_channel.astype(np.int32)
        self.b_channel = self.b_channel.astype(np.int32)
        a_ = self.a_channel - 107
        b_ = self.b_channel - 220
        a = np.sqrt(np.square(a_) + np.square(b_))
        print('a min: ', np.min(a))

        print('a max: ', np.max(a))
        red_mask = a <= r
        rest_mask = a > r
        # create another image making the pixels of the red_mask black and the remaining white
        redness_image = np.zeros_like(self.image_preprocessed)
        redness_image[red_mask] = self.image_preprocessed[red_mask]
        redness_image[rest_mask] = 255
        # Save the image
        save_path = './test_result/brownness_image.jpg'
        cv2.imwrite(save_path, redness_image)

        # gather range info in three channels
        # Gather the red pixels in each channel
        red_pixels_l = self.l_channel[red_mask]
        red_pixels_a = self.a_channel[red_mask]
        red_pixels_b = self.b_channel[red_mask]

        # Calculate the extreme values f red pixels in each channel
        l_min, l_max = np.min(red_pixels_l), np.max(red_pixels_l)
        a_min, a_max = np.min(red_pixels_a), np.max(red_pixels_a)
        b_min, b_max = np.min(red_pixels_b), np.max(red_pixels_b)

        # histogram equalization on l, a and b in each channel

        # lower bound of l is l_min - 20% of the range if it is positive, otherwise 0
        lower_bound_l_equ = l_min - (l_max - l_min) * 0.2 if l_min - (l_max - l_min) * 0.2 > 0 else 0
        # upper bound of l is l_max - 20% of the range if it is less than 255, otherwise 255
        upper_bound_l_equ = l_max - (l_max - l_min) * 0.2 if l_max + (l_max - l_min) * 0.2 < 255 else 255

        self.l_channel[red_mask] = (self.l_channel[red_mask] - l_min) * (
                    (upper_bound_l_equ - lower_bound_l_equ) / (l_max - l_min)) + lower_bound_l_equ
        
        lower_bound_b_equ = b_min + (b_max - b_min) * 0.5
        # upper bound of a is b_max + 20% of the range
        upper_bound_b_equ = b_max + (b_max - b_min) * 1
        self.b_channel[red_mask] = (self.b_channel[red_mask] - b_min) * (
                    (upper_bound_b_equ - lower_bound_b_equ) / (b_max - b_min)) + lower_bound_b_equ

        # for the rest pixels, increase l by 20% and move the point on the ab plane towards the origin by 50%
        self.l_channel[rest_mask] = np.where(
            self.l_channel[rest_mask] * 1.3 < 255,
            self.l_channel[rest_mask] * 1.3,
            255
        )
        self.a_channel[rest_mask] = np.full_like(self.a_channel[rest_mask], 127)
        self.b_channel[rest_mask] = np.full_like(self.b_channel[rest_mask], 127)
        # self.a_channel[rest_mask] = self.a_channel[rest_mask] * 0.5
        # self.b_channel[rest_mask] = self.b_channel[rest_mask] * 0.5

        # turn the channels back to uint8
        self.b_channel = self.a_channel.astype(np.uint8)
        self.l_channel = self.l_channel.astype(np.uint8)
        # turn channel b to all 127
        self.a_channel = np.full_like(self.a_channel, 127).astype(np.uint8)

        if self.l_channel.shape == self.a_channel.shape == self.b_channel.shape:
            # Check if the channels have the same depth (data type)
            if self.l_channel.dtype == self.a_channel.dtype == self.b_channel.dtype:
                # Merge the channels
                self.result_lab = cv2.merge([self.l_channel, self.a_channel, self.b_channel])
            else:
                print("Channel depths (data types) do not match.")
        else:
            print("Channel dimensions do not match.")
        # modified_cielab = cv2.merge([self.l_channel, self.a_channel, self.b_channel])

        # Convert the modified CIELab image back to the original color space
        self.result = cv2.cvtColor(self.result_lab, cv2.COLOR_Lab2RGB)
        self.result = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        # Display the result
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        save_path = './test_result/sample_b.jpg'
        cv2.imwrite(save_path, self.result)
        save_path = './test_result/image_preprocessed.jpg'
        cv2.imwrite(save_path, self.image_preprocessed)
        save_path = './test_result/image_org.jpg'
        cv2.imwrite(save_path, self.image_org)
        # save_path = './test_result/image_lab.jpg'
        # cv2.imwrite(save_path, self.image_lab)

    def run(self):
        self.preprocess_image()
        self.visualize_y()
        self.visualize_redness()
        #self.find_unrepresentable_pixels()

    def find_unrepresentable_pixels(self, threshold=1.0):
        height, width, _ = self.result_lab.shape
        unrepresentable_pixels = []

        for y in range(height):
            for x in range(width):
                lab_color = self.result_lab[y, x]

                # Convert color from LAB to BGR
                bgr_color = cv2.cvtColor(np.uint8([[lab_color]]), cv2.COLOR_LAB2BGR)[0, 0]

                # Convert color from BGR to LAB
                converted_lab_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2LAB)[0, 0]

                # Calculate the Euclidean distance between the original and converted LAB colors
                distance = np.linalg.norm(lab_color - converted_lab_color)

                if distance > threshold:
                    unrepresentable_pixels.append((y, x))


        # Print the number of unrepresentable pixels
        print(f"Number of unrepresentable pixels: {len(unrepresentable_pixels)}")

        for pixel in unrepresentable_pixels:
            y, x, lab_color = pixel[0],pixel[1], self.result_lab[pixel[0], pixel[1]]
            print(f"Pixel at (x={y}, y={x}): LAB color = {lab_color}")

    def test_lab_to_bgr_conversion(self, threshold=1.0):
        # Generate all possible LAB color values
        l_values = np.arange(1, 256)  # L channel values from 0 to 100
        a_values = np.arange(1, 256)  # A channel values from -128 to 127
        b_values = np.arange(1, 256)  # B channel values from -128 to 127

        # Create a grid of LAB color values

        unrepresentable_pixels = []

        for i in l_values:
            for j in a_values:
                for k in b_values:
                    #create a pixel with ijk
                    lab_color = np.array([i,j,k], dtype=np.uint8)
                    # Convert color from LAB to BGR
                    bgr_color = cv2.cvtColor(np.uint8([[lab_color]]), cv2.COLOR_LAB2BGR)[0, 0]

                    # Convert color from BGR to LAB
                    converted_lab_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2LAB)[0, 0]

                    # Calculate the Euclidean distance between the original and converted LAB colors
                    distance = np.linalg.norm(lab_color - converted_lab_color)

                    if distance < threshold:
                        unrepresentable_pixels.append(lab_color)

        # Print the number of unrepresentable pixels
        print(f"Number of unrepresentable pixels: {len(unrepresentable_pixels)}")

        # for pixel in unrepresentable_pixels:
        #
        #     print(pixel)
        # Convert the list of NumPy arrays to a pandas DataFrame
        df = pd.DataFrame(unrepresentable_pixels[:1000000], columns=["L", "A", "B"])

        # Define the Excel file path
        excel_file_path = "output.xlsx"

        # Create a pandas Excel writer
        writer = pd.ExcelWriter(excel_file_path, engine="xlsxwriter")

        # Write the DataFrame to the Excel file
        df.to_excel(writer, index=False)

        # Save the Excel file
        writer.save()

        # Optional: Provide feedback to confirm the successful storage
        print("Data saved to Excel file:", excel_file_path)

        #return unequal_lab_colors





sample_img = cv2.imread('sample/t1.jpg')
cvt = ConvertToRedness(sample_img)
cvt.run()
#cvt.test_lab_to_bgr_conversion()