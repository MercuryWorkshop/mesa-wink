// SPDX-License-Identifier: MIT
package org.mesa.wink;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.rendering.v1.WorldRenderEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.world.World;

public final class Fixture implements ClientModInitializer {
    private int ticks, frames;
    private boolean prepared, ready;
    private final String scene = System.getProperty("wink.fixture.scene", "day");

    @Override public void onInitializeClient() {
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (client.world == null || client.player == null || client.getServer() == null) return;
            if (!prepared) {
                prepared = true;
                var server = client.getServer();
                server.execute(() -> {
                    var manager = server.getCommandManager();
                    var source = server.getCommandSource().withLevel(4);
                    String dimension = scene.equals("nether") ? "minecraft:the_nether" : scene.equals("end") ? "minecraft:the_end" : "minecraft:overworld";
                    String prefix = "execute in " + dimension + " run ";
                    String[] commands = {
                        "gamerule doDaylightCycle false", "gamerule doWeatherCycle false",
                        "gamerule doMobSpawning false", "gamerule randomTickSpeed 0",
                        "gamemode spectator @a",
                        "time set " + (scene.equals("night") ? "18000" : "6000"),
                        "weather " + (scene.equals("rain") ? "rain" : "clear"),
                        prefix + "fill -16 72 -16 16 76 16 minecraft:stone",
                        prefix + "fill -16 77 -16 16 90 16 minecraft:air",
                        prefix + "fill -12 76 -12 12 76 12 minecraft:grass_block",
                        prefix + "fill -9 77 -8 -6 80 -5 minecraft:glass",
                        prefix + "fill -3 77 -8 0 80 -5 minecraft:gold_block",
                        prefix + "fill 3 77 -8 6 80 -5 minecraft:polished_deepslate",
                        prefix + "fill -10 77 0 10 77 5 minecraft:water",
                        prefix + "fill -10 78 5 10 78 5 minecraft:oak_fence",
                        prefix + "setblock 2 78 -3 minecraft:sea_lantern",
                        prefix + "setblock -2 78 -3 minecraft:redstone_lamp[lit=true]",
                        "execute as @a in " + dimension + " run tp @s " +
                            (scene.equals("water") ? "0 76.0 2 180 0" : "0 80 11 180 12")
                    };
                    for (String command : commands) manager.executeWithPrefix(source, command);
                });
            }
            if (++ticks == 180) {
                var expected = scene.equals("nether") ? World.NETHER : scene.equals("end") ? World.END : World.OVERWORLD;
                if (!client.world.getRegistryKey().equals(expected))
                    throw new IllegalStateException("Fixture dimension was not entered: " + scene);
                if (scene.equals("water") && !client.player.isSubmergedInWater())
                    throw new IllegalStateException("Fixture camera is not underwater");
                try {
                    Class<?> api = Class.forName("net.irisshaders.iris.api.v0.IrisApi");
                    Object instance = api.getMethod("getInstance").invoke(null);
                    if (!((Boolean)api.getMethod("isShaderPackInUse").invoke(instance)))
                        throw new IllegalStateException("Iris shader pack fallback");
                } catch (ReflectiveOperationException e) { throw new RuntimeException(e); }
                client.options.hudHidden = true;
                client.setScreen(null);
                System.out.println("WINK_FIXTURE Iris active " + System.getProperty("wink.fixture.pack") + " scene=" + scene);
                ready = true;
            }
        });
        WorldRenderEvents.START.register(context -> {
            if (ready && scene.equals("motion"))
                MinecraftClient.getInstance().player.setYaw(180 + (frames % 32) * 0.2f);
        });
        WorldRenderEvents.END.register(context -> {
            if (!ready) return;
            if (++frames == 96) {
                System.out.println("WINK_FIXTURE minecraft complete frames=" + frames);
                MinecraftClient.getInstance().scheduleStop();
            }
        });
    }
}
