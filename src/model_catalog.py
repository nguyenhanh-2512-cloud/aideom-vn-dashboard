"""Model catalog for VN AIDEOM-VN dashboard.

This module keeps reusable metadata for the 12 exercises.
It is intentionally separated from app.py so the project has a clearer
software structure when submitted to GitHub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ExerciseSpec:
    """Specification of one decision-model exercise."""
    number: int
    title: str
    level: str
    method: str
    data_scope: str
    core_outputs: List[str]
    policy_questions: List[str]


EXERCISES: Dict[int, ExerciseSpec] = {
    1: ExerciseSpec(
        number=1,
        title='Hàm sản xuất Cobb-Douglas mở rộng với AI và số hóa',
        level='Dễ',
        method='Cobb-Douglas, TFP, growth accounting',
        data_scope='Macro 2020-2025',
        core_outputs=['TFP A_t theo năm', 'MAPE dự báo', 'Phân rã tăng trưởng', 'GDP 2030'],
        policy_questions=['TFP tăng hay giảm?', 'D, AI, H đóng góp thế nào?', 'Mục tiêu kinh tế số 30% GDP 2030 có khả thi không?'],
    ),
    2: ExerciseSpec(
        number=2,
        title='Phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số',
        level='Dễ',
        method='Linear programming',
        data_scope='Ngân sách số 2026',
        core_outputs=['Phân bổ tối ưu', 'Giá trị Z*', 'Độ nhạy ngân sách', 'Tác động tăng sàn nhân lực'],
        policy_questions=['Shadow price ngân sách có ý nghĩa gì?', 'R&D hiệu quả cao nhưng vì sao sàn thấp?', 'Tỷ lệ công nghệ chiến lược có khả thi không?'],
    ),
    3: ExerciseSpec(
        number=3,
        title='Tính chỉ số ưu tiên ngành Priority_i',
        level='Dễ',
        method='MCDM, min-max, weighted score',
        data_scope='10 ngành Việt Nam 2024',
        core_outputs=['Ma trận chuẩn hóa', 'Priority từng ngành', 'Top-3 ngành', 'Độ nhạy trọng số AI'],
        policy_questions=['Top ngành có phù hợp định hướng chuyển đổi số không?', 'Khai khoáng năng suất cao nhưng vì sao không ưu tiên?', 'Ai nên quyết định trọng số chính sách?'],
    ),
    4: ExerciseSpec(
        number=4,
        title='Quy hoạch tuyến tính phân bổ ngân sách số ngành-vùng',
        level='Trung bình',
        method='LP với ràng buộc công bằng',
        data_scope='6 vùng và 4 hạng mục',
        core_outputs=['Ma trận phân bổ 6x4', 'Z*', 'Heatmap phân bổ', 'Chi phí công bằng vùng'],
        policy_questions=['Bỏ công bằng vốn chảy về đâu?', 'Trần vùng có phải chính sách phân quyền không?', 'Tây Nguyên nên ưu tiên AI hay H/I trước?'],
    ),
    5: ExerciseSpec(
        number=5,
        title='Quy hoạch nguyên hỗn hợp lựa chọn dự án chuyển đổi số',
        level='Trung bình',
        method='MIP, binary knapsack',
        data_scope='15 dự án 2026-2030',
        core_outputs=['Danh mục dự án chọn', 'Tổng chi phí', 'Tổng NPV', 'NPV/chi phí'],
        policy_questions=['Vì sao Open Data có thể bị bỏ qua?', 'An ninh mạng bắt buộc có làm giảm Z* không?', 'Mô hình hóa cộng hưởng P8-P13 thế nào?'],
    ),
    6: ExerciseSpec(
        number=6,
        title='TOPSIS xếp hạng 6 vùng kinh tế',
        level='Trung bình',
        method='TOPSIS, entropy weight',
        data_scope='6 vùng KT-XH',
        core_outputs=['C_i* expert', 'C_i* entropy', 'Xếp hạng vùng', 'So sánh thứ hạng'],
        policy_questions=['Vùng nào triển khai trung tâm AI trước?', 'Entropy làm vùng nào đổi hạng?', 'Tương quan tiêu chí ảnh hưởng kết quả thế nào?'],
    ),
    7: ExerciseSpec(
        number=7,
        title='Tối ưu đa mục tiêu Pareto với NSGA-II',
        level='Khá khó',
        method='Pareto, multi-objective optimization',
        data_scope='Budget vùng và tham số rủi ro',
        core_outputs=['Tập nghiệm Pareto', 'Scatter 3D', 'Nghiệm thỏa hiệp', 'Chi phí cơ hội'],
        policy_questions=['Đánh đổi tăng trưởng và bao trùm có rõ không?', 'Trọng số mục tiêu nên điều chỉnh thế nào?', 'AI có thay thế quyết định chính trị không?'],
    ),
    8: ExerciseSpec(
        number=8,
        title='Tối ưu động phân bổ liên thời gian 2026-2035',
        level='Khá khó',
        method='Dynamic optimization',
        data_scope='Macro state variables',
        core_outputs=['Quỹ đạo K', 'Quỹ đạo D/AI/H', 'GDP và tiêu dùng', 'Phân tích cú sốc'],
        policy_questions=['Đầu tư nên front-load hay back-load?', 'AI/H theo thời gian nói gì về đào tạo?', 'Hệ số chiết khấu thấp làm kết quả đổi thế nào?'],
    ),
    9: ExerciseSpec(
        number=9,
        title='Tác động AI tới thị trường lao động Việt Nam',
        level='Khá khó',
        method='LP, labor simulation',
        data_scope='8 ngành lao động',
        core_outputs=['x_AI và x_H', 'NetJob từng ngành', 'Retraining capacity', 'Ngành rủi ro'],
        policy_questions=['Ngành nào cần đào tạo lại nhiều nhất?', 'Tài chính-ngân hàng nên dùng chiến lược gì?', 'Tốc độ tự động hóa được chặn bằng ràng buộc nào?'],
    ),
    10: ExerciseSpec(
        number=10,
        title='Quy hoạch ngẫu nhiên hai giai đoạn',
        level='Khó',
        method='Two-stage stochastic programming',
        data_scope='4 kịch bản kinh tế',
        core_outputs=['First-stage x', 'Second-stage y_s', 'Expected objective', 'VSS/EVPI minh họa'],
        policy_questions=['SP đầu tư H nhiều hơn hay ít hơn?', 'VSS dương nói gì về tư duy xác suất?', 'Nhân lực số có vai trò hàng hóa bảo hiểm không?'],
    ),
    11: ExerciseSpec(
        number=11,
        title='Q-learning cho chính sách kinh tế thích nghi',
        level='Khó',
        method='Tabular reinforcement learning',
        data_scope='MDP 81 trạng thái',
        core_outputs=['Q-table', 'Policy π*', 'Learning curve', 'So sánh rule-based'],
        policy_questions=['GDP thấp, D thấp, U cao chọn gì?', 'GDP cao, AI cao, U thấp chọn gì?', 'Tích hợp π* vào quy trình chính sách thế nào?'],
    ),
    12: ExerciseSpec(
        number=12,
        title='Đồ án tích hợp AIDEOM-VN',
        level='Khó',
        method='Integrated dashboard',
        data_scope='M1-M6, 5 scenarios',
        core_outputs=['Bảng KPI 5 kịch bản', 'Dashboard tổng quan', 'Cảnh báo rủi ro', 'Khuyến nghị chính sách'],
        policy_questions=['Kịch bản nào cân bằng nhất?', 'Kết quả 2030 khác nhau ra sao?', 'Dashboard hỗ trợ quyết định nhưng giới hạn ở đâu?'],
    ),
}


SCENARIO_DESCRIPTIONS = {
    "S1 - Truyền thống": "Tập trung vốn vật chất, FDI, hạ tầng truyền thống và xuất khẩu.",
    "S2 - Số hóa nhanh": "Tăng đầu tư chính phủ số, doanh nghiệp số và thanh toán số.",
    "S3 - AI dẫn dắt": "Ưu tiên AI, dữ liệu lớn, bán dẫn và trung tâm dữ liệu.",
    "S4 - Bao trùm số": "Ưu tiên vùng yếu, SME, giáo dục số và nông nghiệp số.",
    "S5 - Tối ưu cân bằng": "Kịch bản tổng hợp từ mô hình AIDEOM-VN.",
}


def get_exercise(number: int) -> ExerciseSpec:
    """Return metadata for one exercise."""
    return EXERCISES[number]


def list_exercises() -> List[ExerciseSpec]:
    """Return all exercise specifications ordered by exercise number."""
    return [EXERCISES[i] for i in sorted(EXERCISES)]


def exercises_by_level(level: str) -> List[ExerciseSpec]:
    """Filter exercises by difficulty level."""
    return [spec for spec in list_exercises() if spec.level == level]


def dashboard_minimum_check() -> dict:
    """Checklist used before submission.

    The dashboard should be checked against the final project requirements:
    data availability, local run, Streamlit deployment, README, requirements file,
    GitHub repository, report, presentation, and demo video.
    """
    return {
        "has_app_py": True,
        "has_requirements": True,
        "has_data_folder": True,
        "has_13_pages": True,
        "has_home_page": True,
        "has_12_exercise_pages": True,
        "has_policy_interpretation": True,
        "needs_report_pdf": True,
        "needs_slides": True,
        "needs_demo_video": True,
    }

FORMULA_LIBRARY = [
    {"name": 'Cobb-Douglas', "formula": 'Y_t = A_t K_t^alpha L_t^beta D_t^gamma AI_t^delta H_t^theta'},
    {"name": 'Growth accounting', "formula": 'ΔlnY = ΔlnA + αΔlnK + βΔlnL + γΔlnD + δΔlnAI + θΔlnH'},
    {"name": 'LP budget', "formula": 'max Z = Σ c_j x_j subject to budget, floor, and strategic-technology constraints'},
    {"name": 'Priority', "formula": 'Priority_i = Σ a_k normalized_indicator_ik with risk reversed or penalized'},
    {"name": 'Regional LP', "formula": 'max Σ_r Σ_j β_jr x_jr subject to regional floor, ceiling, and fairness'},
    {"name": 'MIP', "formula": 'max Σ B_i y_i with y_i ∈ {0,1}, budget, precedence, and exclusion constraints'},
    {"name": 'TOPSIS', "formula": 'C_i* = S_i- / (S_i+ + S_i-)'},
    {"name": 'Pareto', "formula": 'A solution is Pareto-efficient if no objective improves without worsening another objective'},
    {"name": 'Dynamic optimization', "formula": 'max Σ ρ^t U(C_t) subject to state transition equations'},
    {"name": 'NetJob', "formula": 'NetJob_i = NewJob_i + UpgradeJob_i - DisplacedJob_i'},
    {"name": 'Stochastic programming', "formula": 'max first-stage benefit + expected recourse benefit'},
    {"name": 'Q-learning', "formula": "Q(s,a) ← Q(s,a)+η[r+γ max_a Q(s',a)-Q(s,a)]"},
]

POLICY_INTERPRETATION_TEMPLATE = {
    1: [
        'Kết quả của bài 1 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày hàm sản xuất cobb-douglas mở rộng với ai và số hóa, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    2: [
        'Kết quả của bài 2 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày phân bổ ngân sách đơn giản theo 4 hạng mục đầu tư số, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    3: [
        'Kết quả của bài 3 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày tính chỉ số ưu tiên ngành priority_i, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    4: [
        'Kết quả của bài 4 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày quy hoạch tuyến tính phân bổ ngân sách số ngành-vùng, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    5: [
        'Kết quả của bài 5 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày quy hoạch nguyên hỗn hợp lựa chọn dự án chuyển đổi số, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    6: [
        'Kết quả của bài 6 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày topsis xếp hạng 6 vùng kinh tế, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    7: [
        'Kết quả của bài 7 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày tối ưu đa mục tiêu pareto với nsga-ii, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    8: [
        'Kết quả của bài 8 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày tối ưu động phân bổ liên thời gian 2026-2035, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    9: [
        'Kết quả của bài 9 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày tác động ai tới thị trường lao động việt nam, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    10: [
        'Kết quả của bài 10 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày quy hoạch ngẫu nhiên hai giai đoạn, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    11: [
        'Kết quả của bài 11 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày q-learning cho chính sách kinh tế thích nghi, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
    12: [
        'Kết quả của bài 12 cần được đọc như một công cụ hỗ trợ ra quyết định, không phải mệnh lệnh chính sách tự động.',
        'Khi trình bày đồ án tích hợp aideom-vn, nhóm nên nêu rõ giả định, dữ liệu, ràng buộc và giới hạn diễn giải.',
        'Phần thảo luận nên gắn với bối cảnh Việt Nam, năng lực thực thi và đánh đổi giữa tăng trưởng, bao trùm, rủi ro.',
    ],
}

def policy_notes(number: int) -> List[str]:
    """Return suggested policy interpretation notes for one exercise."""
    return POLICY_INTERPRETATION_TEMPLATE[number]

# The following plain dictionaries help students export sections into a report.
REPORT_SECTION_TITLES = {
    1: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    2: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    3: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    4: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    5: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    6: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    7: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    8: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    9: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    10: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    11: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
    12: ['Bối cảnh và mục tiêu', 'Mô hình toán học', 'Dữ liệu và giả định', 'Kết quả tính toán', 'Diễn giải chính sách', 'Giới hạn và hướng mở rộng'],
}

def report_outline(number: int) -> List[str]:
    """Return the suggested report outline for one exercise."""
    return REPORT_SECTION_TITLES[number]

__all__ = [
    'ExerciseSpec', 'EXERCISES', 'SCENARIO_DESCRIPTIONS', 'FORMULA_LIBRARY',
    'get_exercise', 'list_exercises', 'exercises_by_level', 'dashboard_minimum_check',
    'policy_notes', 'report_outline',
]