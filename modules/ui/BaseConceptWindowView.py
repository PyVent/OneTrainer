import fractions
import math

from modules.util.ui.pyside6_i18n import set_localized_text, translate


class BaseConceptWindowView:
    def __init__(self, components):
        self.components = components
        self.bucket_ax = None
        self.text_color = None
        self.canvas = None

    def _update_concept_stats(self, controller):
        #file size
        set_localized_text(self.file_size_preview, "{megabytes} MB",
                           megabytes=int(controller.concept.concept_stats["file_size"]/1048576))
        set_localized_text(self.processing_time, "{seconds} s",
                           seconds=round(controller.concept.concept_stats["processing_time"], 2))

        #directory count
        self.components.set_label_text(self.dir_count_preview, controller.concept.concept_stats["directory_count"])

        #image count
        self.components.set_label_text(self.image_count_preview, controller.concept.concept_stats["image_count"])
        self.components.set_label_text(self.image_count_mask_preview, controller.concept.concept_stats["image_with_mask_count"])
        self.components.set_label_text(self.image_count_caption_preview, controller.concept.concept_stats["image_with_caption_count"])

        #video count
        self.components.set_label_text(self.video_count_preview, controller.concept.concept_stats["video_count"])
        self.components.set_label_text(self.video_count_caption_preview, controller.concept.concept_stats["video_with_caption_count"])

        #mask count
        self.components.set_label_text(self.mask_count_preview, controller.concept.concept_stats["mask_count"])
        self.components.set_label_text(self.mask_count_preview_unpaired, controller.concept.concept_stats["unpaired_masks"])

        #caption count
        if controller.concept.concept_stats["subcaption_count"] > 0:
            self.components.set_label_text(self.caption_count_preview, f'{controller.concept.concept_stats["caption_count"]} ({controller.concept.concept_stats["subcaption_count"]})')
        else:
            self.components.set_label_text(self.caption_count_preview, controller.concept.concept_stats["caption_count"])
        self.components.set_label_text(self.caption_count_preview_unpaired, controller.concept.concept_stats["unpaired_captions"])

        #resolution info
        max_pixels = controller.concept.concept_stats["max_pixels"]
        avg_pixels = controller.concept.concept_stats["avg_pixels"]
        min_pixels = controller.concept.concept_stats["min_pixels"]

        if any(isinstance(x, str) for x in [max_pixels, avg_pixels, min_pixels]) or not sum(controller.concept.concept_stats["aspect_buckets"].values()):
            self.components.set_label_text(self.pixel_max_preview, "-")
            self.components.set_label_text(self.pixel_avg_preview, "-")
            self.components.set_label_text(self.pixel_min_preview, "-")
        else:
            #formatted as (#pixels/1000000) MP, width x height, \n filename
            set_localized_text(self.pixel_max_preview, "{megapixels} MP, {size}\n{name}",
                               megapixels=round(max_pixels[0]/1000000, 2), size=max_pixels[2], name=max_pixels[1])
            set_localized_text(self.pixel_avg_preview, "{megapixels} MP, ~{width}w x {height}h",
                               megapixels=round(avg_pixels/1000000, 2), width=int(math.sqrt(avg_pixels)),
                               height=int(math.sqrt(avg_pixels)))
            set_localized_text(self.pixel_min_preview, "{megapixels} MP, {size}\n{name}",
                               megapixels=round(min_pixels[0]/1000000, 2), size=min_pixels[2], name=min_pixels[1])

        #video length and fps info
        max_length = controller.concept.concept_stats["max_length"]
        avg_length = controller.concept.concept_stats["avg_length"]
        min_length = controller.concept.concept_stats["min_length"]
        max_fps = controller.concept.concept_stats["max_fps"]
        avg_fps = controller.concept.concept_stats["avg_fps"]
        min_fps = controller.concept.concept_stats["min_fps"]

        if any(isinstance(x, str) for x in [max_length, avg_length, min_length]) or controller.concept.concept_stats["video_count"] == 0:   #will be str if adv stats were not taken
            self.components.set_label_text(self.length_max_preview, "-")
            self.components.set_label_text(self.length_avg_preview, "-")
            self.components.set_label_text(self.length_min_preview, "-")
            self.components.set_label_text(self.fps_max_preview, "-")
            self.components.set_label_text(self.fps_avg_preview, "-")
            self.components.set_label_text(self.fps_min_preview, "-")
        else:
            #formatted as (#frames) frames \n filename
            set_localized_text(self.length_max_preview, "{count} frames\n{name}",
                               count=int(max_length[0]), name=max_length[1])
            set_localized_text(self.length_avg_preview, "{count} frames", count=int(avg_length))
            set_localized_text(self.length_min_preview, "{count} frames\n{name}",
                               count=int(min_length[0]), name=min_length[1])
            #formatted as (#fps) fps \n filename
            set_localized_text(self.fps_max_preview, "{count} fps\n{name}", count=int(max_fps[0]), name=max_fps[1])
            set_localized_text(self.fps_avg_preview, "{count} fps", count=int(avg_fps))
            set_localized_text(self.fps_min_preview, "{count} fps\n{name}", count=int(min_fps[0]), name=min_fps[1])

        #caption info
        max_caption_length = controller.concept.concept_stats["max_caption_length"]
        avg_caption_length = controller.concept.concept_stats["avg_caption_length"]
        min_caption_length = controller.concept.concept_stats["min_caption_length"]

        if any(isinstance(x, str) for x in [max_caption_length, avg_caption_length, min_caption_length]) or controller.concept.concept_stats["subcaption_count"] == 0:
            self.components.set_label_text(self.caption_max_preview, "-")
            self.components.set_label_text(self.caption_avg_preview, "-")
            self.components.set_label_text(self.caption_min_preview, "-")
        else:
            #formatted as (#chars) chars, (#words) words, \n filename
            set_localized_text(self.caption_max_preview, "{chars} chars, {words} words\n{name}",
                               chars=max_caption_length[0], words=max_caption_length[2], name=max_caption_length[1])
            set_localized_text(self.caption_avg_preview, "{chars} chars, {words} words",
                               chars=int(avg_caption_length[0]), words=int(avg_caption_length[1]))
            set_localized_text(self.caption_min_preview, "{chars} chars, {words} words\n{name}",
                               chars=min_caption_length[0], words=min_caption_length[2], name=min_caption_length[1])

        #aspect bucketing
        if controller.concept.concept_stats.get("scan_complete") is False:
            return  # Live counters update without redrawing the entire chart per file batch.
        aspect_buckets = controller.concept.concept_stats["aspect_buckets"]
        if len(aspect_buckets) != 0 and max(val for val in aspect_buckets.values()) > 0:    #check aspect_bucket data exists and is not all zero
            min_val = min(val for val in aspect_buckets.values() if val > 0)                #smallest nonzero values
            if max(val for val in aspect_buckets.values()) > min_val:                       #check if any buckets larger than min_val exist - if all images are same aspect then there won't be
                min_val2 = min(val for val in aspect_buckets.values() if (val > 0 and val != min_val))  #second smallest bucket
            else:
                min_val2 = min_val  #if no second smallest bucket exists set to min_val
            min_aspect_buckets = {key: val for key,val in aspect_buckets.items() if val in (min_val, min_val2)}
            min_bucket_str = ""
            for key, val in min_aspect_buckets.items():
                min_bucket_str += translate('aspect {ratio} : {count} img').format(
                    ratio=self.decimal_to_aspect_ratio(key), count=val) + '\n'
            self.components.set_label_text(self.small_bucket_preview, min_bucket_str.strip())
        else:
            self.components.set_label_text(self.small_bucket_preview, "-")

        self.bucket_ax.cla()
        aspects = [str(x) for x in list(aspect_buckets.keys())]
        aspect_ratios = [self.decimal_to_aspect_ratio(x) for x in list(aspect_buckets.keys())]
        counts = list(aspect_buckets.values())
        b = self.bucket_ax.bar(range(len(counts)), counts)
        self.bucket_ax.set_xticks(range(len(counts)), labels=aspect_ratios, rotation=90)
        self.bucket_ax.bar_label(b, color=self.text_color)
        sec = self.bucket_ax.secondary_xaxis(location=-0.1)
        sec.spines["bottom"].set_linewidth(0)
        sec.set_xticks([0, (len(aspects)-1)/2, len(aspects)-1],
                       labels=[translate("Wide"), translate("Square"), translate("Tall")])
        sec.tick_params('x', length=0)
        self.canvas.draw()

    def decimal_to_aspect_ratio(self, value : float):
        #find closest fraction to decimal aspect value and convert to a:b format
        aspect_fraction = fractions.Fraction(value).limit_denominator(16)
        aspect_string = f'{aspect_fraction.denominator}:{aspect_fraction.numerator}'
        return aspect_string

    def _disable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, False)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, False)

    def _enable_scan_buttons(self):
        self.components.set_widget_enabled(self.refresh_basic_stats_button, True)
        self.components.set_widget_enabled(self.refresh_advanced_stats_button, True)

    def _cancel_concept_stats(self, controller):
        controller.cancel_scan_flag.set()
