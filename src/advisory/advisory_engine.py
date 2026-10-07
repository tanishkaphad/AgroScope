"""AgroScope Irrigation Advisory Engine.

Translates ML-predicted water stress (LOW, MODERATE, HIGH) and agro-meteorological
conditions into actionable, tiered irrigation priorities:
- PRIORITY 1 (Urgent): Immediate irrigation required within 24-48 hours
- PRIORITY 2 (Scheduled): Schedule irrigation within 3-5 days
- PRIORITY 3 (Normal): Adequate moisture; routine monitoring
Supports bilingual advisories (English & Marathi) tailored for Maharashtra canal command areas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class IrrigationAdvisoryEngine:
    """Rule-based decision logic translating water stress into trilingual irrigation advisories (EN, MR, HI)."""

    @staticmethod
    def evaluate_cell(
        stress_class: int,
        confidence: float,
        ndwi_mean: float,
        temp_max_c: float = 32.0,
        et0_8day_sum: float = 35.0,
        rainfall_8day: float = 0.0,
        crop_type: Optional[str] = "Command Area Crop",
    ) -> Dict[str, Any]:
        """Generate tailored irrigation priority and recommendations in English, Marathi, and Hindi."""
        # Class 2: HIGH STRESS
        if stress_class == 2:
            if temp_max_c >= 33.0 or et0_8day_sum >= 35.0 or ndwi_mean < 0.02:
                priority = 1
                color = "#dc3545"  # Danger Red
                urgency_en = "Immediate (Next 24-48 Hours)"
                urgency_mr = "तातडीने (पुढील २४-४८ तासांत)"
                urgency_hi = "तुरंत (अगले २४-४८ घंटों में)"

                title_en = "CRITICAL: Immediate Irrigation Required"
                title_mr = "तातडीची सूचना: पुढील २४-४८ तासांत पाणी द्या"
                title_hi = "गंभीर चेतावनी: तुरंत सिंचाई आवश्यक"

                rec_en = (
                    f"Severe canopy water deficit detected with high thermal demand ({temp_max_c:.1f}°C). "
                    f"Leaf moisture is critically depleted (NDWI: {ndwi_mean:.2f}). "
                    f"Immediate canal rotation or tubewell irrigation is urgently required to prevent permanent yield loss."
                )
                rec_mr = (
                    f"पिकात तीव्र पाण्याचा ताण आणि उच्च तापमान ({temp_max_c:.1f}°C) आढळले आहे. "
                    f"पानांमधील ओलावा अत्यंत कमी झाला आहे (NDWI: {ndwi_mean:.2f}). "
                    f"उत्पादनातील नुकसान टाळण्यासाठी पुढील २४ ते ४८ तासांत तातडीने सिंचन करणे गरजेचे आहे."
                )
                rec_hi = (
                    f"फसल में गंभीर जल तनाव और उच्च तापमान ({temp_max_c:.1f}°C) दर्ज किया गया है। "
                    f"पत्तियों में नमी अत्यंत कम हो चुकी है (NDWI: {ndwi_mean:.2f})। "
                    f"उपज में भारी गिरावट रोकने के लिए अगले २४ से ४८ घंटों में तुरंत नहर या नलकूप से सिंचाई करें।"
                )

                actions_en = [
                    "Request canal rotation turn immediately or utilize emergency tubewell",
                    "Irrigate during early morning or late evening to minimize evaporative loss",
                    "Check for localized soil cracking in furrow lines",
                ]
                actions_mr = [
                    "कालवा पाणी वाटप पाळीनुसार तातडीने पाण्याचे नियोजन करा किंवा विहीर/बोअरवेलचा वापर करा",
                    "बाष्पीभवन कमी करण्यासाठी सकाळी लवकर किंवा संध्याकाळी पाणी द्या",
                    "शेतातील सऱ्यांमध्ये भेगा पडल्या आहेत का ते तपासा",
                ]
                actions_hi = [
                    "नहर जल वितरण रोस्टर से तुरंत अपनी बारी लें अथवा आपातकालीन नलकूप का उपयोग करें",
                    "वाष्पीकरण रोकने के लिए सुबह जल्दी या शाम के समय सिंचाई करें",
                    "खेत की क्यारियों में दरारें पड़ने की जांच करें",
                ]
            else:
                priority = 2
                color = "#ffc107"  # Warning Amber
                urgency_en = "Schedule Within 3-5 Days"
                urgency_mr = "पुढील ३-५ दिवसांत नियोजन करा"
                urgency_hi = "अगले ३-५ दिनों में योजना बनाएं"

                title_en = "ELEVATED: Schedule Irrigation Soon"
                title_mr = "सावधान: पुढील ३-५ दिवसांत पाणी देण्याचे नियोजन करा"
                title_hi = "सावधानी: अगले ३-५ दिनों में सिंचाई की योजना बनाएं"

                rec_en = (
                    f"Crop is experiencing water deficit under moderate temperatures ({temp_max_c:.1f}°C). "
                    f"Plan irrigation in the upcoming canal rotation window within 3-5 days."
                )
                rec_mr = (
                    f"पिकास पाण्याचा ताण जाणवत आहे. पुढील ३ ते ५ दिवसांतील कालवा पाळीनुसार सिंचनाचे नियोजन करा."
                )
                rec_hi = (
                    f"फसल में पानी की कमी के लक्षण हैं। अगले ३ से ५ दिनों के भीतर आने वाली नहर बारी में सिंचाई करें।"
                )

                actions_en = [
                    "Confirm your turn in the next canal water distribution roster",
                    "Prepare field channels and furrows for incoming water delivery",
                ]
                actions_mr = [
                    "पुढील कालवा पाणी वाटप पाळीमध्ये तुमची वेळ तपासून घ्या",
                    "पाणी येण्यापूर्वी शेतातील पाट आणि चाऱ्यांची दुरुस्ती करून ठेवा",
                ]
                actions_hi = [
                    "आगामी नहर जल वितरण रोस्टर में अपनी बारी की पुष्टि करें",
                    "पानी आने से पहले खेत की नालियों और क्यारियों की मरम्मत कर लें",
                ]

        # Class 1: MODERATE STRESS
        elif stress_class == 1:
            if rainfall_8day > 5.0:
                priority = 3
                color = "#28a745"
                urgency_en = "Monitor (Rainfall Buffer Available)"
                urgency_mr = "निरीक्षण करा (पावसामुळे ओलावा उपलब्ध)"
                urgency_hi = "निगरानी रखें (बारिश की नमी उपलब्ध)"

                title_en = "MODERATE: Recent Rain Moisture Buffer"
                title_mr = "मध्यम: नुकत्याच झालेल्या पावसामुळे दिलासा"
                title_hi = "मध्यम: हालिया बारिश से नमी उपलब्ध"

                rec_en = (
                    f"Recent rainfall ({rainfall_8day:.1f} mm) provides a temporary soil moisture buffer. "
                    f"Monitor leaf vigor over the next 4-6 days."
                )
                rec_mr = f"नुकत्याच झालेल्या पावसामुळे ({rainfall_8day:.1f} मिमी) तात्पुरता ओलावा आहे. पुढील ४-६ दिवस पिकाचे निरीक्षण करा."
                rec_hi = f"हाल ही में हुई बारिश ({rainfall_8day:.1f} मिमी) से मिट्टी में नमी बनी हुई है। अगले ४-६ दिनों तक फसल की निगरानी करें।"

                actions_en = ["Monitor soil moisture at 15cm depth before applying irrigation"]
                actions_mr = ["पुढील पाणी देण्यापूर्वी जमिनीतील १५ सेमी खोलीवरचा ओलावा तपासा"]
                actions_hi = ["सिंचाई करने से पहले जमीन में १५ सेमी गहराई पर नमी की जांच करें"]
            else:
                priority = 2
                color = "#fd7e14"  # Orange
                urgency_en = "Schedule Within 3-5 Days"
                urgency_mr = "पुढील ३-५ दिवसांत सिंचन आवश्यक"
                urgency_hi = "अगले ३-५ दिनों में सिंचाई आवश्यक"

                title_en = "MODERATE STRESS: Irrigation Window Approaching"
                title_mr = "मध्यम ताण: पुढील ३-५ दिवसांत सिंचन आवश्यक"
                title_hi = "मध्यम तनाव: अगले ३-५ दिनों में पानी दें"

                rec_en = (
                    f"Canopy leaf water is declining (NDWI: {ndwi_mean:.2f}). "
                    f"Soil moisture depletion is in progress. Schedule irrigation within 3-5 days to avoid severe stress."
                )
                rec_mr = (
                    f"पानातील पाण्याचे प्रमाण कमी होत आहे (NDWI: {ndwi_mean:.2f}). "
                    f"तीव्र ताण टाळण्यासाठी पुढील ३ ते ५ दिवसांत सिंचन करा."
                )
                rec_hi = (
                    f"पत्तियों में पानी की मात्रा कम हो रही है (NDWI: {ndwi_mean:.2f})। "
                    f"गंभीर तनाव से बचने के लिए अगले ३ से ५ दिनों के भीतर सिंचाई अवश्य करें।"
                )

                actions_en = [
                    "Inspect field root-zone moisture",
                    "Coordinate with water user association (WUA) for canal release",
                ]
                actions_mr = [
                    "मुळांमधील ओलावा तपासा",
                    "पाणी वापर संस्थेशी (WUA) संपर्क साधून पाळीची वेळ निश्चित करा",
                ]
                actions_hi = [
                    "जड़ क्षेत्र की नमी का निरीक्षण करें",
                    "जल उपभोक्ता संस्था (WUA) से संपर्क कर नहर के पानी का समय तय करें",
                ]

        # Class 0: LOW STRESS
        else:
            priority = 3
            color = "#28a745"  # Success Green
            urgency_en = "Adequate Moisture (No Action Needed)"
            urgency_mr = "पुरेसा ओलावा (पाण्याची गरज नाही)"
            urgency_hi = "पर्याप्त नमी (सिंचाई की आवश्यकता नहीं)"

            title_en = "OPTIMAL: Adequate Crop Hydration"
            title_mr = "उत्तम: पिकात पुरेसा ओलावा उपलब्ध"
            title_hi = "उत्कृष्ट: फसल में पर्याप्त नमी उपलब्ध"

            rec_en = (
                f"Canopy vigor and leaf water content are healthy (NDWI: {ndwi_mean:.2f}). "
                f"No irrigation is needed at present. Preserve canal water quota for later cycles."
            )
            rec_mr = (
                f"पिकाची वाढ आणि पाण्याचे प्रमाण उत्तम आहे (NDWI: {ndwi_mean:.2f}). "
                f"सध्या पाण्याची आवश्यकता नाही. कालव्याचे पाणी पुढील वापरासाठी वाचवा."
            )
            rec_hi = (
                f"फसल की सेहत और पत्तियों में नमी बहुत अच्छी है (NDWI: {ndwi_mean:.2f})। "
                f"वर्तमान में सिंचाई की आवश्यकता नहीं है। नहर का पानी आगामी चक्र के लिए सुरक्षित रखें।"
            )

            actions_en = ["Maintain routine scouting; no irrigation required at this cycle"]
            actions_mr = ["पिकाचे नियमित निरीक्षण ठेवा; या आठवड्यात पाणी देण्याची गरज नाही"]
            actions_hi = ["नियमित निगरानी बनाए रखें; इस सप्ताह पानी की आवश्यकता नहीं है"]

        return {
            "stress_class_code": int(stress_class),
            "stress_class_label": ["LOW", "MODERATE", "HIGH"][int(stress_class)],
            "confidence_pct": round(float(confidence) * 100, 1),
            "priority": priority,
            "priority_label": f"PRIORITY {priority}",
            "color": color,
            "urgency": {
                "en": urgency_en,
                "mr": urgency_mr,
                "hi": urgency_hi,
            },
            "title": {
                "en": title_en,
                "mr": title_mr,
                "hi": title_hi,
            },
            "recommendation": {
                "en": rec_en,
                "mr": rec_mr,
                "hi": rec_hi,
            },
            "actionable_steps": {
                "en": actions_en,
                "mr": actions_mr,
                "hi": actions_hi,
            },
        }

    @staticmethod
    def get_localized_view(advisory: Dict[str, Any], lang: str = "en") -> Dict[str, Any]:
        """Return a localized snapshot for a specific language ('en', 'mr', or 'hi')."""
        l = lang.lower() if lang.lower() in ("en", "mr", "hi") else "en"
        return {
            "stress_class_code": advisory["stress_class_code"],
            "stress_class_label": advisory["stress_class_label"],
            "confidence_pct": advisory["confidence_pct"],
            "priority": advisory["priority"],
            "priority_label": advisory["priority_label"],
            "color": advisory["color"],
            "urgency": advisory["urgency"].get(l, advisory["urgency"]["en"]),
            "title": advisory["title"].get(l, advisory["title"]["en"]),
            "recommendation": advisory["recommendation"].get(l, advisory["recommendation"]["en"]),
            "actionable_steps": advisory["actionable_steps"].get(l, advisory["actionable_steps"]["en"]),
            "language": l,
        }

