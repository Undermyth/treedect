use eframe::egui;
use egui::Color32;
use egui_extras::{Column, TableBuilder};

use crate::panels::global;

pub struct TablePanel {}

impl TablePanel {
    pub fn new() -> Self {
        Self {}
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, global: &mut global::GlobalState) {
        if global.score_table.is_none() {
            return;
        }
        let table = global.score_table.as_ref().unwrap();

        // 定义颜色
        let header_bg_color = Color32::from_rgb(230, 230, 230);
        let row_even_color = Color32::from_rgb(245, 245, 245);
        let row_odd_color = Color32::from_rgb(255, 255, 255);

        TableBuilder::new(ui)
            .column(Column::auto().resizable(true))
            .column(Column::remainder())
            .header(20.0, |mut header| {
                header.col(|ui| {
                    // 绘制header背景色
                    let rect = ui.available_rect_before_wrap();
                    ui.painter().rect_filled(rect, 0.0, header_bg_color);

                    ui.label(egui::RichText::new("Cluster ID").size(14.0).strong());
                });
                header.col(|ui| {
                    // 绘制header背景色
                    let rect = ui.available_rect_before_wrap();
                    ui.painter().rect_filled(rect, 0.0, header_bg_color);

                    ui.label(egui::RichText::new("Score").size(14.0).strong());
                });
            })
            .body(|mut body| {
                for (index, entry) in table.entries.iter().enumerate() {
                    let row_bg_color = if index % 2 == 0 {
                        row_even_color
                    } else {
                        row_odd_color
                    };

                    body.row(20.0, |mut row| {
                        row.col(|ui| {
                            // 绘制row背景色
                            let rect = ui.available_rect_before_wrap();
                            ui.painter().rect_filled(rect, 0.0, row_bg_color);

                            ui.label(format!("{}", entry.id));
                        });
                        row.col(|ui| {
                            // 绘制row背景色
                            let rect = ui.available_rect_before_wrap();
                            ui.painter().rect_filled(rect, 0.0, row_bg_color);

                            ui.label(format!("{:.2}", entry.score));
                        });
                    });
                }
            });
    }
}
